/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: process_stop_unittest.cc
* Date: 26-9-6
************************************************/

// ProcessStop: SIGINT/SIGTERM (and request_stop) must release WaitGroup so
// daemon mains can reach server->stop() and the impl destructor drain.

#include <gtest/gtest.h>

#include <chrono>
#include <csignal>
#include <signal.h>
#include <thread>
#include <unistd.h>

#include "common/process_stop.h"

using mortred::common::ProcessStop;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;

namespace {

long elapsed_ms(steady_clock::time_point t0) {
    return duration_cast<milliseconds>(steady_clock::now() - t0).count();
}

}  // namespace

TEST(process_stop, request_stop_unblocks_wait) {
    ProcessStop stop;
    stop.arm();
    std::thread t([&] {
        std::this_thread::sleep_for(milliseconds(20));
        stop.request_stop();
    });
    const auto t0 = steady_clock::now();
    stop.wait();
    t.join();
    EXPECT_LT(elapsed_ms(t0), 2000);
}

TEST(process_stop, request_stop_is_idempotent) {
    ProcessStop stop;
    stop.arm();
    stop.request_stop();
    stop.request_stop();
    stop.wait();
}

TEST(process_stop, destructor_without_wait_joins) {
    // The destructor must join the signal thread instead of leaving it running
    // or blocking; an armed-and-never-stopped instance is the path main() takes
    // when it returns early.
    const auto t0 = std::chrono::steady_clock::now();
    {
        ProcessStop stop;
        stop.arm();
    }
    EXPECT_LT(elapsed_ms(t0), 1000)
        << "destructor must join the signal thread promptly, not wait for a signal";
}

TEST(process_stop, sigterm_unblocks_wait) {
    ProcessStop stop;
    stop.arm();
    std::thread t([] {
        std::this_thread::sleep_for(milliseconds(50));
        ::kill(::getpid(), SIGTERM);
    });
    const auto t0 = steady_clock::now();
    stop.wait();
    t.join();
    EXPECT_LT(elapsed_ms(t0), 2000);
}

TEST(process_stop, sigint_unblocks_wait) {
    ProcessStop stop;
    stop.arm();
    std::thread t([] {
        std::this_thread::sleep_for(milliseconds(50));
        ::kill(::getpid(), SIGINT);
    });
    const auto t0 = steady_clock::now();
    stop.wait();
    t.join();
    EXPECT_LT(elapsed_ms(t0), 2000);
}
