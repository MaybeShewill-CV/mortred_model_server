/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: worker_pool.h
* Date: 26-9-9
************************************************/

// Worker-instance lease: the blocking queue of model workers plus the
// checkout/checkin protocol, stuck-worker detector and run-time EWMA.
// Not a thread pool. Compute still runs on Workflow go-tasks; this object
// only answers "is a model instance in the queue?".
//
// Callers MUST write metrics / EWMA before checkin: destructor drain waits
// on available_approx() == watermark(), and may destroy metrics after the
// worker is home.

#ifndef MORTRED_SERVER_WORKER_POOL_H
#define MORTRED_SERVER_WORKER_POOL_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>

#include "glog/logging.h"
#include "stl_container/blockingconcurrentqueue.h"

#include "common/time_stamp.h"

namespace jinq {
namespace server {
using jinq::common::Timestamp;

enum class StuckWorkerAction { LOG, EXIT };

inline int64_t worker_monotonic_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename WORKER>
class WorkerPool {
public:
    struct Checkout {
        bool ok = false;
        double wait_ms = 0;
    };
    using NowMs = int64_t (*)();

    explicit WorkerPool(NowMs now = nullptr)
        : _now(now != nullptr ? now : &worker_monotonic_ms) {}

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    void configure_stuck(StuckWorkerAction action, int threshold_times, int timeout_ms) {
        _stuck_action = action;
        _stuck_threshold_times = threshold_times > 0 ? threshold_times : 3;
        _stuck_timeout_ms = timeout_ms;
    }

    void seed_ewma(int64_t ms) {
        _ewma_ms.store(ms > 0 ? ms : 500, std::memory_order_relaxed);
    }

    void adopt(WORKER&& w) {
        _queue.enqueue(std::move(w));
    }

    void commit_watermark(size_t n) {
        _watermark = n;
    }

    Checkout checkout(WORKER& out, std::chrono::milliseconds wait) {
        Checkout result;
        const auto wait_start = Timestamp::now();
        bool got = true;
        if (wait.count() < 0) {
            _queue.wait_dequeue(out);
        } else {
            got = _queue.wait_dequeue_timed(out, wait);
        }
        result.wait_ms = (Timestamp::now() - wait_start) * 1000;
        if (!got) {
            on_wait_timeout();
            result.ok = false;
            return result;
        }
        on_worker_acquired();
        result.ok = true;
        return result;
    }

    void checkin(WORKER&& w) {
        _queue.enqueue(std::move(w));
    }

    void observe_run_ms(int64_t run_ms) {
        int64_t observed = _ewma_ms.load(std::memory_order_relaxed);
        while (true) {
            const double next = static_cast<double>(observed) +
                                0.2 * (static_cast<double>(run_ms) - static_cast<double>(observed));
            const int64_t next_i = static_cast<int64_t>(next);
            if (_ewma_ms.compare_exchange_weak(observed, next_i, std::memory_order_relaxed)) {
                break;
            }
        }
    }

    int64_t ewma_ms() const {
        return _ewma_ms.load(std::memory_order_relaxed);
    }

    size_t available_approx() const {
        return _queue.size_approx();
    }

    size_t watermark() const {
        return _watermark;
    }

    void drain() {
        if (_watermark == 0) {
            return;
        }
        constexpr int k_poll_ms = 5;
        while (available_approx() != _watermark) {
            std::this_thread::sleep_for(std::chrono::milliseconds(k_poll_ms));
        }
    }

private:
    void on_wait_timeout() {
        const int consecutive = _consecutive_wait_timeouts.fetch_add(1) + 1;
        if (consecutive == 1) {
            _first_wait_timeout_ms.store(_now());
        }
        const int64_t first = _first_wait_timeout_ms.load();
        const int64_t span_ms = first > 0 ? _now() - first : 0;
        const int64_t threshold_ms =
            static_cast<int64_t>(_stuck_threshold_times) * _stuck_timeout_ms;
        if (consecutive >= _stuck_threshold_times && span_ms >= threshold_ms) {
            if (_stuck_action == StuckWorkerAction::EXIT) {
                LOG(FATAL) << "worker wait timed out " << consecutive
                           << " times in a row with an empty queue, spanning " << span_ms
                           << " ms (>= " << threshold_ms
                           << " ms): worker stuck, exiting for supervisor restart";
            } else if (consecutive == _stuck_threshold_times) {
                LOG(ERROR) << "worker stuck: " << consecutive
                           << " consecutive full-timeout waits spanning " << span_ms << " ms";
            }
        }
    }

    void on_worker_acquired() {
        _consecutive_wait_timeouts.store(0);
        _first_wait_timeout_ms.store(0);
    }

    moodycamel::BlockingConcurrentQueue<WORKER> _queue;
    size_t _watermark = 0;
    NowMs _now;
    StuckWorkerAction _stuck_action = StuckWorkerAction::LOG;
    int _stuck_threshold_times = 3;
    int _stuck_timeout_ms = 500;
    std::atomic<int> _consecutive_wait_timeouts{0};
    std::atomic<int64_t> _first_wait_timeout_ms{0};
    std::atomic<int64_t> _ewma_ms{500};
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_WORKER_POOL_H
