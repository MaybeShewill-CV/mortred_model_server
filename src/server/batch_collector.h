/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: batch_collector.h
* Date: 26-9-9
************************************************/

// Dynamic batch collector: the only serving-runtime component that owns a
// std::thread. Ownership comments from the original inlined implementation
// are preserved. Worker checkout goes exclusively through WorkerPool.
//
// Pipeline: the collector thread only windows items and owns dispatch into a
// bounded ready deque. Model run happens on GoStarter tasks (Workflow go in
// production). Multiple batches may be in flight up to the worker watermark.
// The collector never runs the model as backpressure when the ready gate is
// full — it waits on a condition variable instead.

#ifndef MORTRED_SERVER_BATCH_COLLECTOR_H
#define MORTRED_SERVER_BATCH_COLLECTOR_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "stl_container/blockingconcurrentqueue.h"

#include "common/status_code.h"
#include "common/time_stamp.h"
#include "server/inference_task.h"
#include "server/item_exec.h"
#include "server/prometheus_metrics.h"
#include "server/worker_pool.h"

namespace jinq {
namespace server {
using jinq::common::StatusCode;
using jinq::common::Timestamp;

/*** One whole multi-item request handed to the batch collector. The state
 * OWNS the request and the per-item result slots; every queued batch_entry
 * holds a shared_ptr to it, so the HTTP reply path may race the runner via
 * the request-level timer — late completions still land safely; reply takes
 * an acquire snapshot of published slots (SME-08 claim-write-publish). */
template <typename MODEL_OUTPUT>
struct BatchRequestState {
    // Per-slot publish protocol (no mutex): empty -> claimed -> published.
    // Only the claim winner may write outputs/item_status; readers copy only
    // after published (acquire).
    static constexpr uint8_t kSlotEmpty = 0;
    static constexpr uint8_t kSlotClaimed = 1;
    static constexpr uint8_t kSlotPublished = 2;

    InferenceTask req;
    std::vector<MODEL_OUTPUT> outputs;
    std::vector<StatusCode> item_status;
    std::unique_ptr<std::atomic<uint8_t>[]> slot_state;
    std::atomic<size_t> completed{0};
    std::function<void()> notify_done;
    std::atomic<int64_t> find_worker_ms{0};
    std::atomic<int64_t> worker_run_ms{0};

    void init(size_t n) {
        outputs.assign(n, MODEL_OUTPUT{});
        item_status.assign(n, StatusCode::MODEL_RUN_TIMEOUT);
        slot_state = std::make_unique<std::atomic<uint8_t>[]>(n);
        for (size_t i = 0; i < n; ++i) {
            slot_state[i].store(kSlotEmpty, std::memory_order_relaxed);
        }
        completed.store(0, std::memory_order_relaxed);
    }

    bool slot_published(size_t idx) const {
        return slot_state && idx < outputs.size() &&
               slot_state[idx].load(std::memory_order_acquire) == kSlotPublished;
    }

    /*** Claim -> write payload -> publish. Losers never touch outputs[idx].
     * notify_done runs exactly once when the last slot publishes. */
    static void write_slot(const std::shared_ptr<BatchRequestState>& state, size_t idx,
                           StatusCode status, MODEL_OUTPUT&& output) {
        if (!state || idx >= state->outputs.size() || !state->slot_state) {
            return;
        }
        uint8_t expected = kSlotEmpty;
        if (!state->slot_state[idx].compare_exchange_strong(
                expected, kSlotClaimed, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return;
        }
        state->outputs[idx] = std::move(output);
        state->item_status[idx] = status;
        state->slot_state[idx].store(kSlotPublished, std::memory_order_release);
        const size_t n = state->outputs.size();
        const size_t prev = state->completed.fetch_add(1, std::memory_order_acq_rel);
        if (prev + 1 == n && state->notify_done) {
            state->notify_done();
        }
    }
};

template <typename WORKER, typename MODEL_OUTPUT>
class BatchCollector {
public:
    using GoStarter = std::function<void(std::function<void()>)>;

    BatchCollector(WorkerPool<WORKER>& pool, PrometheusMetrics& metrics,
                   GoStarter go_starter = {})
        : _pool(pool), _metrics(metrics), _go_starter(std::move(go_starter)) {}

    BatchCollector(const BatchCollector&) = delete;
    BatchCollector& operator=(const BatchCollector&) = delete;

    ~BatchCollector() {
        stop();
    }

    void set_go_starter(GoStarter go_starter) {
        _go_starter = std::move(go_starter);
    }

    void configure(int max_batch_size, int max_batch_delay_ms, int worker_wait_timeout_ms) {
        _max_batch_size = max_batch_size < 1 ? 1 : max_batch_size;
        _max_batch_delay_ms = max_batch_delay_ms < 0 ? 5 : max_batch_delay_ms;
        _worker_wait_timeout_ms = worker_wait_timeout_ms;
    }

    void start() {
        if (_max_batch_size <= 1) {
            return;
        }
        if (_thread.joinable()) {
            return;
        }
        _max_in_flight = std::max<size_t>(1, _pool.watermark());
        _accepting.store(true, std::memory_order_release);
        _running.store(true, std::memory_order_release);
        _thread = std::thread([this]() { batch_loop(); });
    }

    /*** stop must fail queued/ready entries, wait for in-flight exec to finish
     * writing slots, then join — matching "stop batch first" in the
     * orchestrator destructor. SME-08: bump epoch (no new mutex) so in-flight
     * submit TOCTOUs fix up unpublished slots; final drain catches late
     * enqueues after join. */
    void stop() {
        _accepting.store(false, std::memory_order_release);
        _epoch.fetch_add(1, std::memory_order_acq_rel);
        if (_thread.joinable()) {
            {
                std::lock_guard<std::mutex> lock(_ready_mu);
                _running.store(false, std::memory_order_release);
                _ready_cv.notify_all();
            }
            // Join first so the collector cannot race a late ++in_flight after we
            // observe zero (it drains queue+ready on the way out). Then wait for
            // any already-dispatched GoStarter exec to finish write_slot.
            _thread.join();
            wait_in_flight_zero();
        }
        drain_queue_and_ready();
    }

    /*** Enqueue one batch_entry per item. Does not block the caller.
     * SME-08: stamp epoch; after enqueue re-check accepting/epoch and TIMEOUT
     * any unpublished slots if stop raced past the first check (no new lock). */
    void submit(std::shared_ptr<BatchRequestState<MODEL_OUTPUT>> state) {
        if (!state) {
            return;
        }
        const size_t n_items = state->req.item_count();
        const uint64_t epoch_at_entry = _epoch.load(std::memory_order_acquire);
        if (!_accepting.load(std::memory_order_acquire)) {
            fail_unpublished_slots(state);
            return;
        }
        for (size_t idx = 0; idx < n_items; ++idx) {
            _queue.enqueue(std::make_shared<batch_entry>(
                batch_entry{idx, state, epoch_at_entry}));
        }
        if (!_accepting.load(std::memory_order_acquire) ||
            _epoch.load(std::memory_order_acquire) != epoch_at_entry) {
            fail_unpublished_slots(state);
        }
    }

private:
    struct batch_entry {
        size_t item_index = 0;
        std::shared_ptr<BatchRequestState<MODEL_OUTPUT>> owner;
        uint64_t epoch = 0;
    };

    static void fail_unpublished_slots(
        const std::shared_ptr<BatchRequestState<MODEL_OUTPUT>>& state) {
        if (!state) {
            return;
        }
        const size_t n = state->outputs.size();
        for (size_t idx = 0; idx < n; ++idx) {
            BatchRequestState<MODEL_OUTPUT>::write_slot(
                state, idx, StatusCode::MODEL_RUN_TIMEOUT, MODEL_OUTPUT{});
        }
    }

    static bool has_finite_deadline(const InferenceTask& req) {
        return req.deadline != std::chrono::steady_clock::time_point::max();
    }

    static bool past_deadline(const InferenceTask& req) {
        return has_finite_deadline(req) &&
               std::chrono::steady_clock::now() >= req.deadline;
    }

    /*** Remaining ms until deadline; -1 if no finite deadline. */
    static int64_t remaining_deadline_ms(const InferenceTask& req) {
        if (!has_finite_deadline(req)) {
            return -1;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            req.deadline - std::chrono::steady_clock::now())
                            .count();
        return ms < 0 ? 0 : ms;
    }

    /*** Drop stale/expired entries with TIMEOUT; keep live (SME-08 epoch + SME-09 deadline). */
    void filter_live_batch(std::vector<std::shared_ptr<batch_entry>>& batch) {
        if (batch.empty()) {
            return;
        }
        const uint64_t epoch_now = _epoch.load(std::memory_order_acquire);
        const bool accepting = _accepting.load(std::memory_order_acquire);
        std::vector<std::shared_ptr<batch_entry>> live;
        live.reserve(batch.size());
        for (auto& entry : batch) {
            if (!entry || !entry->owner) {
                continue;
            }
            if (!accepting || entry->epoch != epoch_now ||
                past_deadline(entry->owner->req)) {
                BatchRequestState<MODEL_OUTPUT>::write_slot(
                    entry->owner, entry->item_index, StatusCode::MODEL_RUN_TIMEOUT,
                    MODEL_OUTPUT{});
                continue;
            }
            live.push_back(std::move(entry));
        }
        batch.swap(live);
    }

    /*** Checkout wait: min(configured cap, min remaining deadline in batch). */
    std::chrono::milliseconds checkout_wait_for_batch(
        const std::vector<std::shared_ptr<batch_entry>>& batch) const {
        int64_t wait_ms = _worker_wait_timeout_ms > 0 ? _worker_wait_timeout_ms : -1;
        for (const auto& entry : batch) {
            if (!entry || !entry->owner) {
                continue;
            }
            const int64_t rem = remaining_deadline_ms(entry->owner->req);
            if (rem < 0) {
                continue;
            }
            if (wait_ms < 0 || rem < wait_ms) {
                wait_ms = rem;
            }
        }
        if (wait_ms < 0) {
            return std::chrono::milliseconds(-1);
        }
        return std::chrono::milliseconds(wait_ms);
    }

    void batch_loop() {
        std::vector<std::shared_ptr<batch_entry>> batch;
        while (_running.load(std::memory_order_acquire)) {
            std::shared_ptr<batch_entry> first;
            if (!_queue.wait_dequeue_timed(first, std::chrono::milliseconds(100))) {
                continue;
            }
            batch.clear();
            batch.push_back(std::move(first));
            const int64_t window_start = worker_monotonic_ms();
            const int64_t window_deadline = window_start + _max_batch_delay_ms;
            while (batch.size() < static_cast<size_t>(_max_batch_size)) {
                const int remain_ms = static_cast<int>(window_deadline - worker_monotonic_ms());
                if (remain_ms <= 0) {
                    break;
                }
                std::shared_ptr<batch_entry> next;
                if (!_queue.wait_dequeue_timed(next, std::chrono::milliseconds(remain_ms))) {
                    break;
                }
                batch.push_back(std::move(next));
            }
            while (batch.size() < static_cast<size_t>(_max_batch_size)) {
                std::shared_ptr<batch_entry> extra;
                if (!_queue.try_dequeue(extra)) {
                    break;
                }
                batch.push_back(std::move(extra));
            }
            filter_live_batch(batch);
            if (batch.empty()) {
                continue;
            }
            _metrics.observe_batch_size(static_cast<double>(batch.size()));
            const int64_t waited_ms = worker_monotonic_ms() - window_start;
            _metrics.observe_batch_window_wait_ms(
                static_cast<double>(waited_ms < 0 ? 0 : waited_ms));

            {
                std::unique_lock<std::mutex> lock(_ready_mu);
                _ready.push_back(std::move(batch));
            }
            for (;;) {
                dispatch_ready();
                std::unique_lock<std::mutex> lock(_ready_mu);
                const size_t ready_cap = 2 * _max_in_flight;
                if (_ready.size() < ready_cap || !_running.load(std::memory_order_acquire)) {
                    break;
                }
                // Gate full: wait for in-flight completion to free ready slots.
                // Never run the model on the collector thread as backpressure.
                _ready_cv.wait(lock, [this, ready_cap]() {
                    return _ready.size() < ready_cap ||
                           !_running.load(std::memory_order_acquire);
                });
            }
        }
        drain_queue_and_ready();
    }

    void dispatch_ready() {
        for (;;) {
            std::vector<std::shared_ptr<batch_entry>> batch;
            {
                std::unique_lock<std::mutex> lock(_ready_mu);
                // Check _running under the same lock as ++_in_flight so stop()
                // cannot observe in_flight==0 and then race a late increment.
                if (!_running.load(std::memory_order_acquire) || _ready.empty() ||
                    _in_flight >= _max_in_flight) {
                    return;
                }
                batch = std::move(_ready.front());
                _ready.pop_front();
                ++_in_flight;
            }
            start_go(std::move(batch));
        }
    }

    void start_go(std::vector<std::shared_ptr<batch_entry>> batch) {
        auto run = [this, batch = std::move(batch)]() mutable {
            run_batch_and_complete(batch);
        };
        if (_go_starter) {
            _go_starter(std::move(run));
        } else {
            // Empty GoStarter: inline only for unit tests. Production always
            // installs a Workflow go starter before start().
            run();
        }
    }

    void run_batch_and_complete(std::vector<std::shared_ptr<batch_entry>>& batch) {
        const auto finish_in_flight = [this]() {
            {
                std::lock_guard<std::mutex> lock(_ready_mu);
                if (_in_flight > 0) {
                    --_in_flight;
                }
            }
            _ready_cv.notify_all();
        };

        if (batch.empty()) {
            finish_in_flight();
            return;
        }

        // SME-09: align with HTTP timer — do not take a worker for expired work.
        filter_live_batch(batch);
        if (batch.empty()) {
            finish_in_flight();
            return;
        }

        WORKER worker;
        const std::chrono::milliseconds wait = checkout_wait_for_batch(batch);
        const auto ck = _pool.checkout(worker, wait);
        if (!ck.ok) {
            for (const auto& entry : batch) {
                entry->owner->find_worker_ms.store(static_cast<int64_t>(ck.wait_ms),
                                                   std::memory_order_relaxed);
                BatchRequestState<MODEL_OUTPUT>::write_slot(
                    entry->owner, entry->item_index, StatusCode::MODEL_RUN_TIMEOUT,
                    MODEL_OUTPUT{});
            }
            finish_in_flight();
            return;
        }
        _metrics.observe_queue_wait_ms(ck.wait_ms);

        // Deadline may have elapsed while waiting on the pool.
        filter_live_batch(batch);
        if (batch.empty()) {
            _pool.checkin(std::move(worker));
            finish_in_flight();
            return;
        }

        using ModelInput = typename WORKER::element_type::input_type;
        std::vector<ModelInput> inputs;
        inputs.reserve(batch.size());
        for (const auto& entry : batch) {
            const auto& owner = entry->owner;
            inputs.push_back(make_model_input<ModelInput>(
                std::move(owner->req.items[entry->item_index]), owner->req.params.get()));
        }
        std::vector<MODEL_OUTPUT> outputs;
        std::vector<StatusCode> item_status;
        const auto run_start = Timestamp::now();
        const auto status = worker->run_batch(inputs, outputs, item_status);
        const double run_ms = (Timestamp::now() - run_start) * 1000;
        // metrics + EWMA before checkin (pool drain / destructor contract)
        _pool.observe_run_ms(static_cast<int64_t>(run_ms));
        _metrics.observe_inference_duration_ms(run_ms);
        _pool.checkin(std::move(worker));

        for (size_t idx = 0; idx < batch.size(); ++idx) {
            const auto& entry = batch[idx];
            entry->owner->find_worker_ms.store(static_cast<int64_t>(ck.wait_ms),
                                               std::memory_order_relaxed);
            entry->owner->worker_run_ms.store(static_cast<int64_t>(run_ms),
                                              std::memory_order_relaxed);
            const StatusCode entry_status = idx < item_status.size() ? item_status[idx] : status;
            if (entry_status == StatusCode::OK && idx < outputs.size()) {
                BatchRequestState<MODEL_OUTPUT>::write_slot(
                    entry->owner, entry->item_index, entry_status, std::move(outputs[idx]));
            } else {
                BatchRequestState<MODEL_OUTPUT>::write_slot(
                    entry->owner, entry->item_index, entry_status, MODEL_OUTPUT{});
            }
        }
        finish_in_flight();
    }

    void drain_queue_and_ready() {
        std::shared_ptr<batch_entry> pending;
        while (_queue.try_dequeue(pending)) {
            BatchRequestState<MODEL_OUTPUT>::write_slot(
                pending->owner, pending->item_index, StatusCode::MODEL_RUN_TIMEOUT,
                MODEL_OUTPUT{});
        }
        std::deque<std::vector<std::shared_ptr<batch_entry>>> ready;
        {
            std::lock_guard<std::mutex> lock(_ready_mu);
            ready.swap(_ready);
        }
        for (auto& batch : ready) {
            for (const auto& entry : batch) {
                BatchRequestState<MODEL_OUTPUT>::write_slot(
                    entry->owner, entry->item_index, StatusCode::MODEL_RUN_TIMEOUT,
                    MODEL_OUTPUT{});
            }
        }
    }

    void wait_in_flight_zero() {
        std::unique_lock<std::mutex> lock(_ready_mu);
        _ready_cv.wait(lock, [this]() { return _in_flight == 0; });
    }

    WorkerPool<WORKER>& _pool;
    PrometheusMetrics& _metrics;
    GoStarter _go_starter;
    int _max_batch_size = 1;
    int _max_batch_delay_ms = 5;
    int _worker_wait_timeout_ms = 500;
    size_t _max_in_flight = 1;
    moodycamel::BlockingConcurrentQueue<std::shared_ptr<batch_entry>> _queue;
    std::mutex _ready_mu;
    std::condition_variable _ready_cv;
    std::deque<std::vector<std::shared_ptr<batch_entry>>> _ready;
    size_t _in_flight = 0;
    std::atomic<bool> _accepting{false};
    std::atomic<uint64_t> _epoch{0};
    std::atomic<bool> _running{false};
    std::thread _thread;
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_BATCH_COLLECTOR_H
