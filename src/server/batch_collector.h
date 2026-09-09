/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: batch_collector.h
* Date: 26-9-9
************************************************/

// Dynamic batch collector: the only serving-runtime component that owns a
// std::thread. Ownership comments from the original inlined implementation
// are preserved. Worker checkout goes exclusively through WorkerPool.

#ifndef MORTRED_SERVER_BATCH_COLLECTOR_H
#define MORTRED_SERVER_BATCH_COLLECTOR_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
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

template <typename WORKER, typename MODEL_OUTPUT>
class BatchCollector {
public:
    BatchCollector(WorkerPool<WORKER>& pool, PrometheusMetrics& metrics)
        : _pool(pool), _metrics(metrics) {}

    BatchCollector(const BatchCollector&) = delete;
    BatchCollector& operator=(const BatchCollector&) = delete;

    ~BatchCollector() {
        stop();
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
        _running.store(true);
        _thread = std::thread([this]() { batch_loop(); });
    }

    /*** stop must fail queued entries and join, matching "stop batch first"
     * in the orchestrator destructor. */
    void stop() {
        if (!_thread.joinable()) {
            return;
        }
        _running.store(false);
        _thread.join();
    }

    /*** enqueue into the batch queue and wait for the runner's completion */
    void submit_and_wait(InferenceTask* req, InferenceResult<MODEL_OUTPUT>* result) {
        auto state = std::make_shared<request_state>();
        state->req = std::move(*req);
        const size_t n_items = state->req.item_count();
        state->outputs.assign(n_items, MODEL_OUTPUT{});
        // unfinished items keep the timeout default: the aggregate derives the
        // DEADLINE_EXCEEDED_PARTIAL / MODEL_RUN_TIMEOUT semantics from it
        state->item_status.assign(n_items, StatusCode::MODEL_RUN_TIMEOUT);
        for (size_t idx = 0; idx < n_items; ++idx) {
            _queue.enqueue(std::make_shared<batch_entry>(batch_entry{idx, state}));
        }

        const auto wait_start = Timestamp::now();
        std::unique_lock<std::mutex> lock(state->mu);
        bool done = true;
        if (state->req.deadline != std::chrono::steady_clock::time_point::max()) {
            done = state->cv.wait_until(lock, state->req.deadline,
                                        [&state]() { return state->done; });
        } else {
            state->cv.wait(lock, [&state]() { return state->done; });
        }
        const double wait_ms = (Timestamp::now() - wait_start) * 1000;
        result->options = state->req.options;
        result->find_worker_time_consuming = wait_ms;
        result->worker_run_time_consuming = state->worker_run_time_consuming;
        result->task_finished_ts = Timestamp::now().to_format_str();
        if (done) {
            result->item_outputs = std::move(state->outputs);
            result->item_status = std::move(state->item_status);
        } else {
            // deadline: the runner may still be writing, so copy under the lock;
            // then latch done so late completions are dropped
            result->item_outputs = state->outputs;
            result->item_status = state->item_status;
            state->done = true;
        }
        aggregate_item_statuses(result);
    }

private:
    /*** One whole multi-item request handed to the batch collector. The
     * state OWNS the request and the per-item result slots; every queued
     * batch_entry holds a shared_ptr to it, so the requesting go task may
     * time out and disappear while items are still queued or running - the
     * last owner frees the state, no use-after-free on any interleaving. */
    struct request_state {
        InferenceTask req;
        std::mutex mu;
        std::condition_variable cv;
        size_t completed = 0;
        bool done = false;
        std::vector<MODEL_OUTPUT> outputs;
        std::vector<StatusCode> item_status;
        double worker_run_time_consuming = 0;
        double find_worker_time_consuming = 0;
    };

    struct batch_entry {
        size_t item_index = 0;
        std::shared_ptr<request_state> owner;
    };

    void batch_loop() {
        std::vector<std::shared_ptr<batch_entry>> batch;
        while (_running.load()) {
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
            _metrics.observe_batch_size(static_cast<double>(batch.size()));
            const int64_t waited_ms = worker_monotonic_ms() - window_start;
            _metrics.observe_batch_window_wait_ms(
                static_cast<double>(waited_ms < 0 ? 0 : waited_ms));
            process_batch(batch);
        }
        std::shared_ptr<batch_entry> pending;
        while (_queue.try_dequeue(pending)) {
            complete_batch_item(pending, StatusCode::MODEL_RUN_TIMEOUT, MODEL_OUTPUT{}, 0, 0);
        }
    }

    void process_batch(std::vector<std::shared_ptr<batch_entry>>& batch) {
        if (batch.empty()) {
            return;
        }
        WORKER worker;
        const std::chrono::milliseconds wait = _worker_wait_timeout_ms > 0
            ? std::chrono::milliseconds(_worker_wait_timeout_ms)
            : std::chrono::milliseconds(-1);
        const auto ck = _pool.checkout(worker, wait);
        if (!ck.ok) {
            for (const auto& entry : batch) {
                complete_batch_item(entry, StatusCode::MODEL_RUN_TIMEOUT, MODEL_OUTPUT{}, 0,
                                    ck.wait_ms);
            }
            return;
        }
        _metrics.observe_queue_wait_ms(ck.wait_ms);

        using ModelInput = typename WORKER::element_type::input_type;
        std::vector<ModelInput> inputs;
        inputs.reserve(batch.size());
        for (const auto& entry : batch) {
            const auto& owner = entry->owner;
            // an item whose request deadline already passed still runs: the
            // requester has woken, late results land in the abandoned state
            inputs.push_back(make_model_input<ModelInput>(
                std::move(owner->req.items[entry->item_index]), owner->req.params.get()));
        }
        std::vector<MODEL_OUTPUT> outputs;
        std::vector<StatusCode> item_status;
        const auto run_start = Timestamp::now();
        const auto status = worker->run_batch(inputs, outputs, item_status);
        const double run_ms = (Timestamp::now() - run_start) * 1000;
        _pool.observe_run_ms(static_cast<int64_t>(run_ms));
        _metrics.observe_inference_duration_ms(run_ms);
        _pool.checkin(std::move(worker));

        for (size_t idx = 0; idx < batch.size(); ++idx) {
            const StatusCode entry_status = idx < item_status.size() ? item_status[idx] : status;
            if (entry_status == StatusCode::OK && idx < outputs.size()) {
                complete_batch_item(batch[idx], entry_status, std::move(outputs[idx]), run_ms,
                                    ck.wait_ms);
            } else {
                complete_batch_item(batch[idx], entry_status, MODEL_OUTPUT{}, run_ms, ck.wait_ms);
            }
        }
    }

    void complete_batch_item(const std::shared_ptr<batch_entry>& entry, StatusCode status,
                             MODEL_OUTPUT&& output, double run_ms, double wait_ms) {
        const auto& owner = entry->owner;
        std::lock_guard<std::mutex> lock(owner->mu);
        if (owner->done) {
            // shutdown raced the runner, or the requester already timed out:
            // late results land in the abandoned state and are dropped
            return;
        }
        owner->outputs[entry->item_index] = std::move(output);
        owner->item_status[entry->item_index] = status;
        owner->worker_run_time_consuming = run_ms;
        owner->find_worker_time_consuming = wait_ms;
        if (++owner->completed >= owner->req.item_count()) {
            owner->done = true;
            owner->cv.notify_all();
        }
    }

    WorkerPool<WORKER>& _pool;
    PrometheusMetrics& _metrics;
    int _max_batch_size = 1;
    int _max_batch_delay_ms = 5;
    int _worker_wait_timeout_ms = 500;
    moodycamel::BlockingConcurrentQueue<std::shared_ptr<batch_entry>> _queue;
    std::atomic<bool> _running{false};
    std::thread _thread;
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_BATCH_COLLECTOR_H
