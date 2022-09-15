/* @file compress.cc
 *
 * Licensed under the MIT License <https://opensource.org/licenses/MIT>.
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2021 Zachary Parker
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <filesystem>
#include <iostream>

#include <fcntl.h>
#include <getopt.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <nvcomp/lz4.hpp>
#include <nvcomp.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

#include <draft/util/Util.hh>
#include <draft/util/Reader.hh>
#include <draft/util/Writer.hh>
#include <draft/util/TaskPool.hh>
#include <draft/util/ThreadExecutor.hh>
#include "Cmd.hh"

namespace {

using namespace draft::util;

using namespace std::chrono_literals;

using Clock = std::chrono::steady_clock;

struct CompressOptions
{
    std::filesystem::path inPath{ };
    std::filesystem::path outPath{ };
    size_t blockSize{1u << 20};
    size_t chunkSize{1u << 16};
};

struct CudaError: public std::runtime_error
{
    explicit CudaError(const cudaError_t &err):
        std::runtime_error(cudaGetErrorString(err))
    {
    }
};

CUfileHandle_t cuFileRegister(const ScopedFd &fd)
{
    auto cuDesc = CUfileDescr_t{ };
    cuDesc.handle.fd = fd.get();
    cuDesc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    auto cuHandle = CUfileHandle_t{ };

    if (auto stat = cuFileHandleRegister(&cuHandle, &cuDesc);
        stat.err != CU_FILE_SUCCESS)
    {
        throw std::runtime_error("draft.compress: unable to register file");
    }

    return cuHandle;
}

class CudaBuffer
{
public:
    CudaBuffer() = default;

    explicit CudaBuffer(std::size_t size)
    {
        cudaMalloc(&data_, size);
        if (auto err = cudaGetLastError(); err != cudaSuccess)
            throw CudaError(err);

        size_ = size;

        fileBuf_ = false;
    }

    ~CudaBuffer() noexcept
    {
        if (fileBuf_)
            cuFileBufDeregister(data_);

        cudaFree(data_);
    }

    uint8_t *data() noexcept { return data_; }

    void fileBufRegister()
    {
        if (auto stat = cuFileBufRegister(data_, size_, 0);
            stat.err != CU_FILE_SUCCESS)
        {
            throw std::runtime_error("draft.compress: unable to register file buffer.");
        }
        fileBuf_ = true;
    }

private:
    uint8_t *data_{ };
    std::size_t size_{ };
    bool fileBuf_{ };
};

class Compressor
{
public:
    using Buffer = BufferPool::Buffer;

    Compressor(BufQueue &queue, BufQueue &compressQueue, size_t chunkSize, size_t blockSize):
        queue_{&queue},
        compressQueue_{&compressQueue},
        chunkSize_{chunkSize}
    {
        stream_ = cudaStream_t{ };
        cudaStreamCreate(&stream_);
        if (auto err = cudaGetLastError(); err != cudaSuccess)
            throw CudaError(err);

        nvcompManager_ = std::make_unique<nvcomp::LZ4Manager>(chunkSize_, NVCOMP_TYPE_CHAR, stream_);
        compressConfig_ = std::make_unique<nvcomp::CompressionConfig>(nvcompManager_->configure_compression(blockSize));

        pool_ = BufferPool::make(compressConfig_->max_compressed_buffer_size, 5);
    }

    bool runOnce(std::stop_token stopToken)
    {
        while (auto desc = queue_->get(100ms))
        {
            const auto len = compress(stopToken, std::move(*desc));
        }

        return !stopToken.stop_requested();
    }

private:
    size_t compress(std::stop_token stopToken, BDesc desc)
    {
        auto gpuInBuf = CudaBuffer(desc.len);
        cudaMemcpy(gpuInBuf.data(), desc.buf->uint8Data(), desc.len, cudaMemcpyHostToDevice);

        auto gpuScratchBuf = CudaBuffer(nvcompManager_->get_required_scratch_buffer_size());
        nvcompManager_->set_scratch_buffer(gpuScratchBuf.data());

        auto gpuOutBuf = CudaBuffer(compressConfig_->max_compressed_buffer_size);

        std::chrono::steady_clock::time_point compressBegin = std::chrono::steady_clock::now();
        nvcompManager_->compress(gpuInBuf.data(), gpuOutBuf.data(), *compressConfig_);
        std::chrono::steady_clock::time_point compressEnd = std::chrono::steady_clock::now();
        spdlog::trace("compression time (us): {}", std::chrono::duration_cast<std::chrono::microseconds>(compressEnd - compressBegin).count());

        if (auto stat = compressConfig_->get_status(); *stat > 0)
        {
            spdlog::error("draft.compress: compression failed:\n"
                          "  nvcompStatus {}"
                          ,  *stat);
            return 0;
        }

        size_t outLen = nvcompManager_->get_compressed_output_size(gpuOutBuf.data());
        auto outBuf = std::make_shared<Buffer>(pool_->get());
        cudaMemcpy(outBuf->uint8Data(), gpuOutBuf.data(), outLen, cudaMemcpyDeviceToHost);

        while (!stopToken.stop_requested() &&
            !compressQueue_->put({outBuf, 1u, desc.offset, outLen}, 100ms))
        {
        }

        return outLen;
    }

    BufferPoolPtr pool_{ };
    BufQueue *queue_{ };
    BufQueue *compressQueue_{ };

    cudaStream_t stream_{ };
    std::unique_ptr<nvcomp::LZ4Manager> nvcompManager_{ };
    std::unique_ptr<nvcomp::CompressionConfig> compressConfig_{ };

    size_t chunkSize_{ };


};

CompressOptions parseOptions(int argc, char **argv)
{
    constexpr const char *shortOpts = "b:c:i:o:h";
    constexpr const struct option longOpts[] = {
        {"block-size", required_argument, nullptr, 'b'},
        {"chunk-size", required_argument, nullptr, 'c'},
        {"in-path", required_argument, nullptr, 'i'},
        {"out-path", required_argument, nullptr, 'o'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    auto subArgc = argc - 1;
    auto subArgv = argv + 1;

    const auto usage = [argv] {
            std::cout << fmt::format(
                "usage: {} compress [-b <block size>][-c <chunk size>][-h] -i <in-file> -o <out-file>\n"
                , ::basename(argv[0]));
        };

    auto opts = CompressOptions{ };

    for (int c = 0; (c = getopt_long(subArgc, subArgv, shortOpts, longOpts, 0)) >= 0; )
    {
        switch (c)
        {
            case 'b':
            {
                size_t pos{ };
                opts.blockSize = std::stoul(optarg, &pos);

                if (pos != std::string(optarg).size())
                {
                    spdlog::error("invalid block size value: {}", optarg);
                    std::exit(1);
                }

                break;
            }
            case 'c':
            {
                size_t pos{ };
                opts.chunkSize = std::stoul(optarg, &pos);

                if (pos != std::string(optarg).size())
                {
                    spdlog::error("invalid chunk size value: {}", optarg);
                    std::exit(1);
                }

                break;
            }
            case 'i':
            {
                opts.inPath = optarg;
                break;
            }
            case 'o':
            {
                opts.outPath = optarg;
                break;
            }
            case 'h':
                usage();
                std::exit(0);
            case '?':
                std::exit(1);
            default:
                break;
        }
    }

    if (optind < subArgc)
    {
        spdlog::error("trailing args..");
        std::exit(1);
    }

    return opts;
}

void compress(const CompressOptions &opts)
{
    auto outFd = ScopedFd{open(opts.outPath.c_str(), O_WRONLY | O_DIRECT)};
    auto cuOutHandle = cuFileRegister(outFd);

    auto inFd = ScopedFd{open(opts.inPath.c_str(), O_RDONLY | O_DIRECT)};
    auto cuInHandle = cuFileRegister(inFd);

    cudaSetDevice(0);
    if (auto err = cudaGetLastError(); err != cudaSuccess)
        throw CudaError(err);

    auto stream = cudaStream_t{ };
    cudaStreamCreate(&stream);
    if (auto err = cudaGetLastError(); err != cudaSuccess)
        throw CudaError(err);

    auto gpuInBuf = CudaBuffer(opts.blockSize);
    gpuInBuf.fileBufRegister();

    int64_t fileReadMicrosecs = 0;
    int64_t compressMicrosecs = 0;

    size_t outOffset = 0;

    size_t fileSize = std::filesystem::file_size(opts.inPath);

    std::chrono::steady_clock::time_point fullCompressBegin = std::chrono::steady_clock::now();
    for (size_t inOffset = 0; inOffset < fileSize; )
    {
        std::chrono::steady_clock::time_point fileReadBegin = std::chrono::steady_clock::now();
        auto inLen = cuFileRead(cuInHandle, gpuInBuf.data(), opts.blockSize, static_cast<off_t>(inOffset), 0);
        std::chrono::steady_clock::time_point fileReadEnd = std::chrono::steady_clock::now();
        fileReadMicrosecs += std::chrono::duration_cast<std::chrono::microseconds>(fileReadEnd - fileReadBegin).count();

        if (inLen < 0)
        {
            spdlog::error("cuFileRead returned {}", inLen);
            break;
        }

        if (!inLen)
        {
            spdlog::error("cuFileRead returned 0 - ending transfer.");
            break;
        }

        auto nvcompManager = nvcomp::LZ4Manager{opts.chunkSize, NVCOMP_TYPE_CHAR, stream};
        auto compressConfig = nvcompManager.configure_compression(opts.blockSize);

        auto gpuScratchBuf = CudaBuffer(nvcompManager.get_required_scratch_buffer_size());
        nvcompManager.set_scratch_buffer(gpuScratchBuf.data());

        auto gpuOutBuf = CudaBuffer(compressConfig.max_compressed_buffer_size);

        std::chrono::steady_clock::time_point compressBegin = std::chrono::steady_clock::now();
        nvcompManager.compress(gpuInBuf.data(), gpuOutBuf.data(), compressConfig);
        std::chrono::steady_clock::time_point compressEnd = std::chrono::steady_clock::now();
        compressMicrosecs += std::chrono::duration_cast<std::chrono::microseconds>(compressEnd - compressBegin).count();

        if (auto stat = compressConfig.get_status(); *stat > 0)
        {
            spdlog::error("draft.compress: compression failed:\n"
                          "  nvcompStatus {}"
                          ,  *stat);
            cudaFree(gpuOutBuf.data());
            break;
        }

        size_t outLen = nvcompManager.get_compressed_output_size(gpuOutBuf.data());

        if (outLen > inLen)
        {
            spdlog::warn("nvcomp compression failed - output larger than input\n"
                         "  is file already compressed?\n"
                         "  output > input -> {} > {}"
                         ,  outLen
                         ,  inLen);
        }

        spdlog::trace("compression info:\n"
                      "  in size:           {} B\n"
                      "  out size:          {} B\n"
                      "  max out size:      {} B\n"
                      "  compression ratio: {} %"
                      ,  inLen
                      ,  outLen
                      ,  compressConfig.max_compressed_buffer_size
                      ,  static_cast<double>(outLen) / static_cast<double>(inLen));

        cuFileWrite(cuOutHandle, gpuOutBuf.data(), outLen, static_cast<off_t>(outOffset), 0);

        inOffset += static_cast<size_t>(inLen);
        outOffset += static_cast<size_t>(outLen);
    }
    std::chrono::steady_clock::time_point fullCompressEnd = std::chrono::steady_clock::now();
    auto fullCompressMicrosecs = std::chrono::duration_cast<std::chrono::microseconds>(fullCompressEnd - fullCompressBegin).count();

    double microsecPerSec = 1'000'000.0;
    double bytePerGbyte = 1'000'000'000.0;

    size_t outFileSize = std::filesystem::file_size(opts.outPath);

    spdlog::info("compression ratio:       {}", static_cast<double>(outFileSize) / static_cast<double>(fileSize));
    spdlog::info("file read avg. GB/s:     {}"
                 , (fileSize / bytePerGbyte) / (fileReadMicrosecs / microsecPerSec));
    spdlog::info("compress avg. GB/s:      {}"
                 , (fileSize / bytePerGbyte) / (compressMicrosecs / microsecPerSec));
    spdlog::info("full compress avg. GB/s: {}"
                 , (fileSize / bytePerGbyte) / (fullCompressMicrosecs / microsecPerSec));
}

} // namespace

namespace draft::cmd {

int nvcompress(int argc, char **argv)
{
    if (argc < 2)
        std::exit(1);

    auto opts = parseOptions(argc, argv);

    auto inFd = std::make_shared<ScopedFd>(open(opts.inPath.c_str(), O_RDONLY | O_DIRECT));

    auto outFd = std::make_shared<ScopedFd>(open(opts.outPath.c_str(), O_WRONLY | O_DIRECT));
    auto rawOutFd = outFd->get();
    if (rawOutFd < 0)
    {
        spdlog::error("unable to open file '{}': {}"
                      , opts.outPath.c_str()
                      , std::strerror(errno));
    }

    auto pool = BufferPool::make(opts.blockSize, 100);
    auto queue = WaitQueue<BDesc>{ };
    queue.setSizeLimit(100);

    auto compressQueue = WaitQueue<BDesc>{ };
    compressQueue.setSizeLimit(10);

    auto compressors = std::vector<Compressor>{ };
    for (size_t i = 0; i < 100; ++i)
    {
        auto compressor = Compressor{queue, compressQueue, opts.chunkSize, opts.blockSize};
        compressors.push_back(std::move(compressor));
    }

    size_t fileSize = std::filesystem::file_size(opts.inPath);

    auto readExec = TaskPool{ };
    readExec.resize(1);
    readExec.setQueueSizeLimit(50);

    auto readResults = std::vector<std::future<int>>{ };
    const auto deadline = Clock::now() + 50ms;
    while (!readExec.cancelled() && Clock::now() < deadline)
    {
        auto reader = Reader(inFd, 0u, {0, fileSize}, pool, queue);

        if (auto future = readExec.launch(std::move(reader)))
        {
            readResults.push_back(std::move(*future));
            break;
        }
    }

    auto compressExec = ThreadExecutor{ };
    compressExec.add(std::move(compressors), ThreadExecutor::Options::DoFinalize);

    auto fileMap = FdMap{ };
    fileMap.insert({1u, rawOutFd});
    auto writeExec = ThreadExecutor{ };
    writeExec.add(Writer(std::move(fileMap), compressQueue), ThreadExecutor::Options::DoFinalize);

    while (true)
    {
        for (auto &r : readResults)
        {
            if (r.valid() && r.wait_for(0ns) == std::future_status::ready)
                r.get();
        }

        std::erase_if(readResults, [](const auto &r) { return !r.valid(); });

        compressExec.runOnce();

        writeExec.runOnce();

        if (readResults.empty())
        {
            compressExec.cancel();
            bool stat = compressExec.finished();
            spdlog::debug("{}", stat);
            if (!stat)
            {
                break;
            }
        }

        writeExec.runOnce();

        std::this_thread::sleep_for(50ms);
    }

    return 0;
}

}
