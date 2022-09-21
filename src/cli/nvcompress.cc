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
#include <nvcomp/bitcomp.hpp>
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
    bool cudaFileIO{false};
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
        compressQueue_{&compressQueue}
    {
        stream_ = cudaStream_t{ };
        cudaStreamCreate(&stream_);
        if (auto err = cudaGetLastError(); err != cudaSuccess)
            throw CudaError(err);

        nvcompManager_ = std::make_unique<nvcomp::BitcompManager>(NVCOMP_TYPE_CHAR, 0, stream_);
        compressConfig_ = std::make_unique<nvcomp::CompressionConfig>(nvcompManager_->configure_compression(blockSize));

        gpuInBuf_ = std::make_unique<CudaBuffer>(blockSize);

        gpuScratchBuf_ = std::make_unique<CudaBuffer>(nvcompManager_->get_required_scratch_buffer_size());
        nvcompManager_->set_scratch_buffer(gpuScratchBuf_->data());

        gpuOutBuf_ = std::make_unique<CudaBuffer>(compressConfig_->max_compressed_buffer_size);

        pool_ = BufferPool::make(compressConfig_->max_compressed_buffer_size, 100);

        alignPool_ = BufferPool::make(4096, 100);

        alignBuf_ = std::make_shared<Buffer>(alignPool_->get());
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
        cudaMemcpy(gpuInBuf_->data(), desc.buf->uint8Data(), desc.len, cudaMemcpyHostToDevice);

        nvcompManager_->compress(gpuInBuf_->data(), gpuOutBuf_->data(), *compressConfig_);

        if (auto stat = compressConfig_->get_status(); *stat > 0)
        {
            spdlog::error("draft.compress: compression failed:\n"
                          "  nvcompStatus {}"
                          ,  *stat);
            return 0;
        }

        size_t outLen = nvcompManager_->get_compressed_output_size(gpuOutBuf_->data());

        if (outLen > desc.len)
        {
            spdlog::warn("draft.compress: nvcomp compression failed - output larger than input\n"
                         "  is file already compressed?\n"
                         "  output > input -> {} > {}"
                         ,  outLen
                         ,  desc.len);
        }

        auto outBuf = std::make_shared<Buffer>(pool_->get());

        cudaMemcpy(outBuf->uint8Data() + alignOffset_,
                   gpuOutBuf_->data(),
                   outLen,
                   cudaMemcpyDeviceToHost);

        if (alignOffset_ > 0 && alignBuf_)
        {
            std::memcpy(outBuf->uint8Data(), alignBuf_->uint8Data(), alignOffset_);
            outLen += alignOffset_;
            spdlog::trace("draft.compress: added previous align offset {} to outLen"
                          ,  alignOffset_);
        }

        spdlog::trace("draft.compress: compression info:\n"
                      "  in size:      {} B\n"
                      "  out size:     {} B\n"
                      "  max out size: {} B\n"
                      "  comp. ratio:  {}"
                      ,  desc.len
                      ,  outLen
                      ,  compressConfig_->max_compressed_buffer_size
                      ,  static_cast<double>(desc.len) / static_cast<double>(outLen));

        auto alignMult = static_cast<double>(outLen) / 4096.0;
        alignOffset_ = static_cast<size_t>((alignMult - std::floor(alignMult)) * 4096.0);
        size_t alignLen = outLen - alignOffset_;

        double newAlignMult = static_cast<double>(alignLen) / 4096.0;
        spdlog::trace("draft.compress: compression alignment:\n"
                      "  out len:             {}\n"
                      "  align offset:        {}\n"
                      "  align len:           {}\n"
                      "  out len / 4096.0:    {}\n"
                      "  align len / 4096.0:  {}"
                      ,  outLen
                      ,  alignOffset_
                      ,  alignLen
                      ,  alignMult
                      ,  newAlignMult);

        if (alignLen > compressConfig_->max_compressed_buffer_size)
            throw std::runtime_error("nvcomp aligned buffer is greater than buffer size");

        spdlog::trace("draft.compress: offset into output file: {}", outOffset_);
        while (!stopToken.stop_requested() &&
            !compressQueue_->put({outBuf, 1u, outOffset_, alignLen}, 100ms))
        {
        }

        std::memcpy(alignBuf_->uint8Data(), outBuf->uint8Data() + alignLen, alignOffset_);

        outOffset_ += alignLen;

        return outLen;
    }

    BufferPoolPtr pool_{ };
    BufferPoolPtr alignPool_{ };
    BufQueue *queue_{ };
    BufQueue *compressQueue_{ };

    cudaStream_t stream_{ };
    std::unique_ptr<nvcomp::nvcompManagerBase> nvcompManager_{ };
    std::unique_ptr<nvcomp::CompressionConfig> compressConfig_{ };

    std::unique_ptr<CudaBuffer> gpuInBuf_{ };
    std::unique_ptr<CudaBuffer> gpuScratchBuf_{ };
    std::unique_ptr<CudaBuffer> gpuOutBuf_{ };

    std::shared_ptr<Buffer> alignBuf_{ };
    size_t alignOffset_{ };

    size_t outOffset_{ };
};

class CompSession
{
public:
    CompSession(const CompressOptions &opts)
    {
        readPool_ = BufferPool::make(opts.blockSize, 100);
        readQueue_.setSizeLimit(100);

        readExec_.resize(1);
        readExec_.setQueueSizeLimit(10);

        auto compressors = std::vector<Compressor>{ };
        for (size_t i = 0; i < 1; ++i)
        {
            auto compressor = Compressor{readQueue_, writeQueue_, opts.chunkSize, opts.blockSize};
            compressors.push_back(std::move(compressor));
        }
        compressExec_.add(std::move(compressors), ThreadExecutor::Options::DoFinalize);

        writeQueue_.setSizeLimit(100);
    }

    ~CompSession() noexcept
    {
        finish();
    }

    void start(const std::filesystem::path &inPath, const std::filesystem::path &outPath)
    {
        auto inFd = std::make_shared<ScopedFd>(open(inPath.c_str(), O_RDONLY | O_DIRECT));
        auto rawInFd = inFd->get();
        if (rawInFd < 0)
        {
            spdlog::error("unable to open file '{}': {}"
                          , inPath.c_str()
                          , std::strerror(errno));
            throw std::runtime_error("unable to open file for reading");
        }
        auto outFd = std::make_shared<ScopedFd>(open(outPath.c_str(), O_WRONLY | O_DIRECT));
        auto rawOutFd = outFd->get();
        if (rawOutFd < 0)
        {
            spdlog::error("unable to open file '{}': {}"
                          , outPath.c_str()
                          , std::strerror(errno));
            throw std::runtime_error("unable to open file for writing");
        }

        size_t fileSize = std::filesystem::file_size(inPath);

        const auto deadline = Clock::now() + 50ms;
        while (!readExec_.cancelled() && Clock::now() < deadline)
        {
            const auto rateDeadline = Clock::now() + 1ms;

            auto reader = Reader(inFd, 0u, {0, fileSize}, readPool_, readQueue_);

            if (auto future = readExec_.launch(std::move(reader)))
            {
                readResults_.push_back(std::move(*future));
                break;
            }

            std::this_thread::sleep_until(rateDeadline);
        }

        auto fileMap = FdMap{ };
        fileMap.insert({1u, rawOutFd});
        writeExec_.add(Writer(std::move(fileMap), writeQueue_), ThreadExecutor::Options::DoFinalize);

        targetFds_.push_back(std::move(inFd));
        targetFds_.push_back(std::move(outFd));
    }

    bool runOnce()
    {
        for (auto &r : readResults_)
        {
            if (r.valid() && r.wait_for(0ns) == std::future_status::ready)
                r.get();
        }

        std::erase_if(readResults_, [](const auto &r) { return !r.valid(); });

        compressExec_.runOnce();

        writeExec_.runOnce();

        if (readResults_.empty())
        {
            writeExec_.waitFinished();
            compressExec_.waitFinished();
            return false;
        }

        return true;
    }

    void finish() noexcept
    {
        readExec_.cancel();
        compressExec_.cancel();
        writeExec_.cancel();
    }

private:
    std::shared_ptr<BufferPool> readPool_;
    WaitQueue<BDesc> readQueue_;
    TaskPool readExec_;
    std::vector<std::future<int>> readResults_;

    WaitQueue<BDesc> writeQueue_;
    ThreadExecutor writeExec_;

    ThreadExecutor compressExec_;

    std::vector<std::shared_ptr<ScopedFd>> targetFds_;
};

CompressOptions parseOptions(int argc, char **argv)
{
    constexpr const char *shortOpts = "b:c:i:o:th";
    constexpr const struct option longOpts[] = {
        {"block-size", required_argument, nullptr, 'b'},
        {"chunk-size", required_argument, nullptr, 'c'},
        {"in-path", required_argument, nullptr, 'i'},
        {"out-path", required_argument, nullptr, 'o'},
        {"cuda-file-io", no_argument, nullptr, 't'},
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
            case 't':
            {
                opts.cudaFileIO = true;
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

void cudaIOCompress(const CompressOptions &opts)
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

    auto nvcompManager = nvcomp::BitcompManager{NVCOMP_TYPE_CHAR, 0, stream};
    auto compressConfig = nvcompManager.configure_compression(opts.blockSize);

    auto gpuScratchBuf = CudaBuffer(nvcompManager.get_required_scratch_buffer_size());
    nvcompManager.set_scratch_buffer(gpuScratchBuf.data());

    auto gpuOutBuf = CudaBuffer(compressConfig.max_compressed_buffer_size);

    auto gpuInBuf = CudaBuffer(opts.blockSize);
    gpuInBuf.fileBufRegister();

    size_t outOffset = 0;

    size_t fileSize = std::filesystem::file_size(opts.inPath);

    for (size_t inOffset = 0; inOffset < fileSize; )
    {
        auto inLen = cuFileRead(cuInHandle, gpuInBuf.data(), opts.blockSize, static_cast<off_t>(inOffset), 0);

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

        nvcompManager.compress(gpuInBuf.data(), gpuOutBuf.data(), compressConfig);

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
            spdlog::warn("draft.compress: nvcomp compression failed - output larger than input\n"
                         "  is file already compressed?\n"
                         "  output > input -> {} > {}"
                         ,  outLen
                         ,  inLen);
        }

        spdlog::trace("draft.compress: compression info:\n"
                      "  in size:           {} B\n"
                      "  out size:          {} B\n"
                      "  max out size:      {} B\n"
                      "  compression ratio: {}"
                      ,  inLen
                      ,  outLen
                      ,  compressConfig.max_compressed_buffer_size
                      ,  static_cast<double>(inLen) / static_cast<double>(outLen));

        cuFileWrite(cuOutHandle, gpuOutBuf.data(), outLen, static_cast<off_t>(outOffset), 0);

        inOffset += static_cast<size_t>(inLen);
        outOffset += static_cast<size_t>(outLen);
    }

    size_t outFileSize = std::filesystem::file_size(opts.outPath);

    spdlog::info("compression ratio: {}"
                 , static_cast<double>(fileSize) / static_cast<double>(outFileSize));
}

void compress(const CompressOptions &opts)
{
    auto compSession = CompSession(opts);

    compSession.start(opts.inPath, opts.outPath);

    auto deadline = Clock::now();
    while(compSession.runOnce())
    {
        std::this_thread::sleep_until(deadline);

        deadline = Clock::now() + 10ms;
    }

    size_t fileSize = std::filesystem::file_size(opts.inPath);
    size_t outFileSize = std::filesystem::file_size(opts.outPath);

    spdlog::info("compression ratio: {}"
                 , static_cast<double>(fileSize) / static_cast<double>(outFileSize));
}

} // namespace

namespace draft::cmd {

int nvcompress(int argc, char **argv)
{
    if (argc < 2)
        std::exit(1);

    auto opts = parseOptions(argc, argv);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (opts.cudaFileIO)
        cudaIOCompress(opts);
    else
        compress(opts);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1'000'000.0;

    spdlog::debug("elapsed time: {} s", elapsedTime);

    return 0;
}

}
