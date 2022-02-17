/**
 * @file PollSet.hh
 *
 * Licensed under the MIT License <https://opensource.org/licenses/MIT>.
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Zachary Parker
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

#ifndef __DRAFT_UTIL_POLL_SET_HH__
#define __DRAFT_UTIL_POLL_SET_HH__

#include <functional>
#include <system_error>
#include <unordered_map>

#include <sys/epoll.h>

#include "ScopedFd.hh"

namespace draft::util {

class PollSet
{
public:
    using Callback = std::function<bool(unsigned)>;
    using EventCallback = std::function<void(const std::vector<epoll_event> &)>;

    PollSet();

    int add(int fd, unsigned events, Callback &&cb = [](unsigned){ return true; });
    int remove(int fd);
    int waitOnce(int tmoMs);
    int waitOnce(int tmoMs, const EventCallback &cb);

    bool empty() const noexcept
    {
        return members_.empty();
    }

private:
    struct Member
    {
        Callback callback;
    };

    ScopedFd epollFd_{ };
    std::unordered_map<int, Member> members_{ };
};

}

#endif
