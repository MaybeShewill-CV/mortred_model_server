/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: process_stop.h
* Date: 26-9-6
************************************************/

// Process-lifetime stop latch for Workflow HTTP daemons.
//
// Workflow's WaitGroup is the right hang-point for main(), but done() is not
// async-signal-safe, so this type never calls it from a signal handler.
// Instead it mirrors ProcessSupervisor: block SIGINT/SIGTERM on the calling
// thread (and every thread created afterwards), then a sigwait thread turns
// the first of those signals into an idempotent WaitGroup::done().
//
// Header-only and workflow-dependent: do not add it to libcommon (that
// library stays workflow-free for tests-only CI). Include only from binaries
// that already link vendored::workflow.

#ifndef MORTRED_COMMON_PROCESS_STOP_H
#define MORTRED_COMMON_PROCESS_STOP_H

#include <atomic>
#include <cerrno>
#include <ctime>
#include <pthread.h>
#include <signal.h>
#include <thread>

#include <workflow/WFFacilities.h>

namespace jinq {
namespace common {

class ProcessStop {
  public:
    ProcessStop() = default;

    ProcessStop(const ProcessStop&) = delete;
    ProcessStop& operator=(const ProcessStop&) = delete;

    ~ProcessStop() {
        _threads_stop.store(true, std::memory_order_release);
        request_stop();
        if (_waiter.joinable()) {
            _waiter.join();
        }
        if (_armed.load(std::memory_order_acquire)) {
            pthread_sigmask(SIG_SETMASK, &_old_mask, nullptr);
        }
    }

    /*** Block SIGINT/SIGTERM, ignore SIGPIPE, start the sigwait thread.
     * Call once on the main thread before spawning glog/Workflow threads so
     * they inherit the mask. */
    void arm() {
        bool expected = false;
        if (!_armed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }
        ::signal(SIGPIPE, SIG_IGN);
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGINT);
        sigaddset(&set, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &set, &_old_mask);
        _waiter = std::thread([this] { signal_loop(); });
    }

    void wait() {
        _wg.wait();
    }

    void request_stop() {
        if (!_released.exchange(true, std::memory_order_acq_rel)) {
            _wg.done();
        }
    }

  private:
    void signal_loop() {
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGINT);
        sigaddset(&set, SIGTERM);
        while (!_threads_stop.load(std::memory_order_acquire)) {
            timespec ts{0, 200 * 1000 * 1000};
            siginfo_t info{};
            const int rc = ::sigtimedwait(&set, &info, &ts);
            if (rc == SIGINT || rc == SIGTERM) {
                request_stop();
                return;
            }
            if (rc < 0 && errno != EAGAIN) {
                continue;
            }
        }
    }

    WFFacilities::WaitGroup _wg{1};
    std::atomic<bool> _released{false};
    std::atomic<bool> _threads_stop{false};
    std::atomic<bool> _armed{false};
    sigset_t _old_mask{};
    std::thread _waiter;
};

}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_PROCESS_STOP_H
