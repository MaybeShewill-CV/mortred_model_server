/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: sync_request_graph.h
* Date: 26-9-10
************************************************/

// Sync inference HTTP graph (P0-timeout). One control flow for N>=1.
//
// Unique reply: a named WFCounterTask(1) sits on the HTTP series. Compute
// is detached (create_go_task + start, not on the series). A request-level
// WFTimerTask also count_by_name's the waiter. Extra counts after the first
// are no-ops (fast N=1 may finish before timer->start()).
//
// Per item: ordinary go tasks, chained from the SUCCESS callback
// (start_item(k+1)). One worker checkout for the whole request; checkin
// after the run() that still holds the lease returns — HTTP 504 / PARTIAL
// may already have been sent.
//
// Publish protocol (no mutex on the in-flight slot):
//   SUCCESS callback writes slot[k], then published.store(k+1, release).
//   reply_cb: replied.exchange(true) once; n = published.load(acquire);
//   copy only [0, n) into a snapshot; pad [n, N) TIMEOUT without reading
//   slot[n].
//
// Batch (max_batch_size > 1): one detached go calling do_work /
// submit_and_wait. No request-level timer — wait_until already wakes at T,
// and an outer timer racing that wait would 504 with published==0 and drop
// collector partials. Hang-past-deadline stays the collector's problem.

#ifndef MORTRED_SERVER_SYNC_REQUEST_GRAPH_H
#define MORTRED_SERVER_SYNC_REQUEST_GRAPH_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "common/status_code.h"
#include "server/inference_task.h"
#include "server/item_exec.h"

namespace protocol {
class HttpResponse;
}

namespace jinq {
namespace server {
using jinq::common::StatusCode;

inline std::string make_sync_waiter_name(const std::string& task_id) {
    static std::atomic<uint64_t> seq{0};
    return "mortred.syncwait." + task_id + "." +
           std::to_string(seq.fetch_add(1, std::memory_order_relaxed));
}

inline std::chrono::milliseconds checkout_wait_for(const InferenceTask& req) {
    if (req.deadline == std::chrono::steady_clock::time_point::max()) {
        return std::chrono::milliseconds(-1);
    }
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                         req.deadline - std::chrono::steady_clock::now())
                         .count();
    if (remaining < 0) {
        remaining = 0;
    }
    return std::chrono::milliseconds(remaining);
}

template <typename MODEL_OUTPUT>
inline InferenceResult<MODEL_OUTPUT> assemble_published(
    const InferenceResult<MODEL_OUTPUT>& src, size_t published, size_t n_items) {
    InferenceResult<MODEL_OUTPUT> out;
    out.options = src.options;
    out.find_worker_time_consuming = src.find_worker_time_consuming;
    out.worker_run_time_consuming = src.worker_run_time_consuming;
    out.task_finished_ts = src.task_finished_ts;
    out.item_status.assign(n_items, StatusCode::MODEL_RUN_TIMEOUT);
    out.item_outputs.assign(n_items, MODEL_OUTPUT{});
    const size_t n = std::min(published, n_items);
    for (size_t i = 0; i < n; ++i) {
        out.item_status[i] = src.item_status[i];
        out.item_outputs[i] = src.item_outputs[i];
    }
    aggregate_item_statuses(&out);
    return out;
}

template <typename WORKER, typename MODEL_OUTPUT>
struct SyncRequestState {
    struct ItemScratch {
        bool ran = false;
        StatusCode status = StatusCode::MODEL_RUN_TIMEOUT;
        MODEL_OUTPUT output{};
        double run_ms = 0;
    };

    InferenceTask req;
    InferenceResult<MODEL_OUTPUT> result;
    protocol::HttpResponse* resp = nullptr;
    std::string waiter;
    std::string task_id;
    size_t n_items = 0;
    std::atomic<size_t> published{0};
    std::atomic<bool> replied{false};
    std::atomic<bool> checkout_failed{false};
    std::atomic<bool> has_worker{false};
    std::atomic<int64_t> find_worker_ms{0};
    std::atomic<int64_t> worker_run_ms{0};
    WORKER worker{};
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_SYNC_REQUEST_GRAPH_H
