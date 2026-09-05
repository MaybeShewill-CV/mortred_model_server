/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: async_job_table.h
* Date: 26-8-23
************************************************/

// AsyncJobTable: the async-job ledger of the model server layer.
//
// Positioning: BaseAiServerImpl stays the single execution orchestrator (it
// owns HTTP parsing, the worker pool shared by the sync and async paths, and
// response serialization). AsyncJobTable is a subordinate component (has-a)
// that owns ONLY the bookkeeping of async jobs: identity, admission, the
// state machine, retention (TTL + LRU) and wait/notify. InferenceTask /
// InferenceResult live in inference_task.h (shared with the sync path).
//
// Concurrency contract (see docs/async-job-table.md):
//   - job state: std::atomic<AsyncJobState>; terminal checks (eviction,
//     polling predicates) read it without taking any lock
//   - result / error / completed_at: guarded by ONE mutex per job; the same
//     mutex guards the condition variable, so every transition notifies
//     inside the critical section - a lost wakeup is impossible by
//     construction
//   - queue depth: std::atomic<int>; admission is a CAS loop, making the
//     queue-full check and the increment one atomic step (no TOCTOU window)
//   - the terminal transition is the ONLY place the depth is decremented,
//     and it is guarded to happen exactly once per job
//   - the table mutex protects only the id map and the LRU deque; it never
//     guards job fields, so map operations never race with transitions

#ifndef MORTRED_MODEL_SERVER_ASYNC_JOB_TABLE_H
#define MORTRED_MODEL_SERVER_ASYNC_JOB_TABLE_H

#include <algorithm>
#include <chrono>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "common/status_code.h"
#include "server/inference_task.h"

namespace jinq {
namespace server {

using jinq::common::StatusCode;

/*** Async job lifecycle. Terminal states are >= DONE. */
enum class AsyncJobState { PENDING = 0, RUNNING = 1, DONE = 2, FAILED = 3, TIMEOUT = 4 };

inline bool is_async_terminal(AsyncJobState state) {
    return state >= AsyncJobState::DONE;
}

inline const char* async_state_str(AsyncJobState state) {
    switch (state) {
        case AsyncJobState::PENDING: return "pending";
        case AsyncJobState::RUNNING: return "running";
        case AsyncJobState::DONE: return "done";
        case AsyncJobState::FAILED: return "failed";
        case AsyncJobState::TIMEOUT: return "timeout";
    }
    return "unknown";
}

// monotonic clock in milliseconds (immune to wall-clock adjustments)
inline int64_t async_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename MODEL_OUTPUT>
class AsyncJobTable {
  public:
    struct Config {
        int max_queue = 16;      // admission bound (depth must stay < max_queue)
        int job_ttl_ms = 300000; // completed-job retention window
        int max_completed = 100; // LRU bound on completed jobs
    };

    enum class SubmitStatus { ACCEPTED, QUEUE_FULL };

    struct SubmitResult {
        SubmitStatus status = SubmitStatus::QUEUE_FULL;
        std::string job_id;  // empty when rejected
    };

    /*** Cheap consistent view for status polling. */
    struct Snapshot {
        std::string id;
        AsyncJobState state = AsyncJobState::PENDING;
        std::string error;           // non-empty on FAILED / TIMEOUT
        int64_t submitted_at_ms = 0;
        int64_t completed_at_ms = 0; // 0 while non-terminal
    };

    enum class ResultStatus { NOT_FOUND, NOT_DONE, READY };

    struct ResultOutcome {
        ResultStatus status = ResultStatus::NOT_FOUND;
        AsyncJobState state = AsyncJobState::PENDING; // NOT_DONE: current state
        std::string task_id;                          // READY: request id echo
        InferenceResult<MODEL_OUTPUT> value;                // READY: copied result
    };

    /*** Set admission/retention config. Call once during server init, before serving. */
    void configure(Config cfg) {
        _m_cfg = cfg;
    }

    const Config& config() const {
        return _m_cfg;
    }

    /***
     * Admit a job (atomic CAS against max_queue), assign a unique id and
     * store the request. The runner later retrieves it via take_request().
     */
    SubmitResult submit(InferenceTask req) {
        if (!try_admit()) {
            return {SubmitStatus::QUEUE_FULL, ""};
        }
        auto job = std::make_shared<Job>();
        job->id = generate_job_id();
        job->req = std::move(req);
        job->submitted_at_ms = async_now_ms();
        {
            std::lock_guard<std::mutex> lock(_m_table_mu);
            _m_jobs[job->id] = job;
            _m_lru.push_back(job->id);
            evict_expired_locked();
        }
        return {SubmitStatus::ACCEPTED, job->id};
    }

    /*** PENDING -> RUNNING. False if the job is missing or not pending. */
    bool transition_running(const std::string& id) {
        auto job = find(id);
        if (job == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(job->mu);
        if (job->state.load() != AsyncJobState::PENDING) {
            return false;
        }
        job->state.store(AsyncJobState::RUNNING);
        job->cv.notify_all();
        return true;
    }

    /*** Terminal DONE with a result. Exactly-once; the only success-path decrement. */
    bool finish(const std::string& id, InferenceResult<MODEL_OUTPUT> result) {
        auto job = find(id);
        if (job == nullptr) {
            return false;
        }
        return transition_terminal(job, AsyncJobState::DONE, "", std::move(result));
    }

    /*** Terminal FAILED with an error message. */
    bool fail(const std::string& id, const std::string& error) {
        auto job = find(id);
        if (job == nullptr) {
            return false;
        }
        return transition_terminal(job, AsyncJobState::FAILED, error, std::nullopt);
    }

    /*** Terminal TIMEOUT with an error message. */
    bool timeout(const std::string& id, const std::string& error) {
        auto job = find(id);
        if (job == nullptr) {
            return false;
        }
        return transition_terminal(job, AsyncJobState::TIMEOUT, error, std::nullopt);
    }

    /***
     * Hand the stored request to the runner: task_id is copied, the payload
     * is moved out (large base64 image). The /jobs/{id}/result request-id
     * echo keeps working because only the payload leaves the job.
     */
    std::optional<InferenceTask> take_request(const std::string& id) {
        auto job = find(id);
        if (job == nullptr) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(job->mu);
        InferenceTask out;
        out.task_id = job->req.task_id;
        out.items = std::move(job->req.items);
        out.params = std::move(job->req.params);
        out.options = job->req.options;
        out.deadline = job->req.deadline;
        return out;
    }

    /*** Consistent view for GET /jobs/{id}. */
    std::optional<Snapshot> snapshot(const std::string& id) {
        auto job = find(id);
        if (job == nullptr) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(job->mu);
        Snapshot snap;
        snap.id = job->id;
        snap.state = job->state.load();
        snap.error = job->error;
        snap.submitted_at_ms = job->submitted_at_ms;
        snap.completed_at_ms = job->completed_at_ms;
        return snap;
    }

    /***
     * Long-poll until the state differs from `initial` or reaches a terminal
     * state, at most `timeout_ms` (<= 0 waits indefinitely). Returns the
     * final snapshot, or nullopt if the job disappeared (evicted).
     * HTTP GET /jobs/{id}/wait does not call this: it hangs a named Workflow
     * counter on the HTTP series instead. Unit tests still use this CV path.
     */
    std::optional<Snapshot> wait(const std::string& id, AsyncJobState initial, int timeout_ms) {
        auto job = find(id);
        if (job == nullptr) {
            return std::nullopt;
        }
        std::unique_lock<std::mutex> lock(job->mu);
        const auto changed = [&job, initial]() {
            const auto s = job->state.load();
            return s != initial || is_async_terminal(s);
        };
        if (timeout_ms > 0) {
            job->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), changed);
        } else {
            job->cv.wait(lock, changed);
        }
        Snapshot snap;
        snap.id = job->id;
        snap.state = job->state.load();
        snap.error = job->error;
        snap.submitted_at_ms = job->submitted_at_ms;
        snap.completed_at_ms = job->completed_at_ms;
        return snap;
    }

    /*** Result lookup for GET /jobs/{id}/result; repeatable until retention ends. */
    ResultOutcome take_result(const std::string& id) {
        auto job = find(id);
        if (job == nullptr) {
            return {};
        }
        std::lock_guard<std::mutex> lock(job->mu);
        ResultOutcome out;
        out.state = job->state.load();
        if (out.state != AsyncJobState::DONE) {
            out.status = ResultStatus::NOT_DONE;
            return out;
        }
        out.status = ResultStatus::READY;
        out.task_id = job->req.task_id;
        out.value = job->result;  // copy: the endpoint stays repeatable
        return out;
    }

    /*** Current admission depth (pending + running). Lock-free. */
    int queue_depth() const {
        return _m_depth.load();
    }

  private:
    struct Job {
        std::string id;
        InferenceTask req;  // payload moved out once by the runner
        int64_t submitted_at_ms = 0;
        std::atomic<AsyncJobState> state{AsyncJobState::PENDING};
        std::mutex mu;     // guards result/error/completed_at + cv
        std::condition_variable cv;
        InferenceResult<MODEL_OUTPUT> result; // valid when state == DONE
        std::string error;              // non-empty on FAILED / TIMEOUT
        // atomic: written under mu but TSAN flags it as a data race when an
        // evicted job's heap block is reused by a new allocation (mutex
        // identity is lost across free/realloc); the mutex already orders the
        // accesses, atomic silences the false positive
        std::atomic<int64_t> completed_at_ms{0};
    };

    static std::string generate_job_id() {
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        static std::atomic<uint64_t> seq{0};
        char buf[48] = {0};
        std::snprintf(buf, sizeof(buf), "job_%016llx_%06u",
                      static_cast<unsigned long long>(now),
                      static_cast<unsigned>(seq.fetch_add(1)));
        return std::string(buf);
    }

    /*** CAS admission: the queue-full check and the increment are one atomic step. */
    bool try_admit() {
        int expected = _m_depth.load();
        while (expected < _m_cfg.max_queue) {
            if (_m_depth.compare_exchange_weak(expected, expected + 1)) {
                return true;
            }
        }
        return false;
    }

    std::shared_ptr<Job> find(const std::string& id) {
        std::lock_guard<std::mutex> lock(_m_table_mu);
        const auto it = _m_jobs.find(id);
        return it == _m_jobs.end() ? nullptr : it->second;
    }

    /***
     * The single terminal transition: exactly-once guarded, stores the
     * payload fields, publishes the state, notifies under the SAME mutex and
     * decrements the depth exactly here.
     */
    bool transition_terminal(const std::shared_ptr<Job>& job,
                             AsyncJobState terminal,
                             const std::string& error,
                             std::optional<InferenceResult<MODEL_OUTPUT>> result) {
        std::lock_guard<std::mutex> lock(job->mu);
        if (is_async_terminal(job->state.load())) {
            return false;  // exactly-once: a second transition is a no-op
        }
        if (result.has_value()) {
            job->result = std::move(*result);
        }
        job->error = error;
        job->completed_at_ms = async_now_ms();
        job->state.store(terminal);
        job->cv.notify_all();
        _m_depth.fetch_sub(1);
        return true;
    }

    /*** TTL + LRU eviction of terminal jobs (caller holds _m_table_mu). */
    void evict_expired_locked() {
        const int64_t now = async_now_ms();
        // TTL: remove terminal jobs past their retention window. completed_at
        // is guarded by the per-job mutex; the lock order is always
        // table -> job, never the reverse, so this cannot deadlock.
        for (auto it = _m_jobs.begin(); it != _m_jobs.end();) {
            const auto& job = it->second;
            if (!is_async_terminal(job->state.load())) {
                ++it;
                continue;
            }
            int64_t completed = 0;
            {
                std::lock_guard<std::mutex> lock(job->mu);
                completed = job->completed_at_ms;
            }
            if (completed > 0 && now - completed > _m_cfg.job_ttl_ms) {
                it = _m_jobs.erase(it);
            } else {
                ++it;
            }
        }
        // LRU: remove oldest terminal jobs beyond max_completed
        int completed_count = 0;
        for (const auto& [id, job] : _m_jobs) {
            (void)id;
            if (is_async_terminal(job->state.load())) {
                ++completed_count;
            }
        }
        while (completed_count > _m_cfg.max_completed && !_m_lru.empty()) {
            const std::string& oldest_id = _m_lru.front();
            const auto it = _m_jobs.find(oldest_id);
            if (it != _m_jobs.end() && is_async_terminal(it->second->state.load())) {
                _m_jobs.erase(it);
                --completed_count;
            }
            _m_lru.pop_front();
        }
        // prune LRU deque entries that no longer exist in the map
        _m_lru.erase(std::remove_if(_m_lru.begin(), _m_lru.end(),
                                    [this](const std::string& id) {
                                        return _m_jobs.find(id) == _m_jobs.end();
                                    }),
                     _m_lru.end());
    }

    Config _m_cfg;
    std::atomic<int> _m_depth{0};
    std::mutex _m_table_mu;
    std::unordered_map<std::string, std::shared_ptr<Job>> _m_jobs;
    std::deque<std::string> _m_lru;  // oldest first
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_MODEL_SERVER_ASYNC_JOB_TABLE_H
