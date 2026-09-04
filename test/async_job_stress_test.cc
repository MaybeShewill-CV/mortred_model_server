/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: async_job_stress_test.cc
* Date: 26-8-23
************************************************/

// Concurrency stress test for the PRODUCTION AsyncJobTable. Registered with
// the "sanitizer" ctest label: the CI sanitizer job runs it (plus
// async_job_unittest) under -fsanitize=thread. It hammers every public entry
// concurrently - submit / transition / finish / snapshot / wait / take_result
// - and asserts the table invariants at the end: depth returns to zero,
// every accepted job reaches a terminal state exactly once, and ids stay
// unique. Under the old inline implementation this workload triggers TSAN
// reports on job->state / job->error / queue_depth reads.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "models/io/common_input.h"
#include "server/async_job_table.h"

namespace {

using jinq::common::StatusCode;
using jinq::server::AsyncJobState;
using jinq::server::AsyncJobTable;
using jinq::server::is_async_terminal;
using jinq::server::InferenceTask;

struct TrivialOutput {
    int value = 0;
};

using Table = AsyncJobTable<TrivialOutput>;
using Result = jinq::server::InferenceResult<TrivialOutput>;

InferenceTask make_req(const std::string& id, const std::string& payload) {
    InferenceTask req;
    req.task_id = id;
    req.items.push_back({jinq::models::io_define::common_io::byte_source::origin_kind::base64_text, payload});
    return req;
}

Result make_result(int value) {
    Result result;
    result.model_run_status = StatusCode::OK;
    result.worker_run_time_consuming = 0.5;
    result.item_status.assign(1, StatusCode::OK);
    result.item_outputs.resize(1);
    result.item_outputs[0].value = value;
    return result;
}

/*** Minimal blocking hand-off queue between submitters and runners. */
struct IdQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> ids;
    bool closed = false;

    void push(const std::string& id) {
        {
            std::lock_guard<std::mutex> lock(mu);
            ids.push_back(id);
        }
        cv.notify_all();
    }

    std::optional<std::string> pop() {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this]() { return closed || !ids.empty(); });
        if (ids.empty()) {
            return std::nullopt;
        }
        std::string id = std::move(ids.front());
        ids.pop_front();
        return id;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu);
            closed = true;
        }
        cv.notify_all();
    }
};

}  // namespace

TEST(async_job_stress, concurrent_lifecycle_preserves_invariants) {
    Table table;
    Table::Config cfg;
    cfg.max_queue = 32;
    cfg.job_ttl_ms = 300000;
    cfg.max_completed = 500;  // exercises concurrent LRU eviction too
    table.configure(cfg);

    IdQueue work;
    std::atomic<int> accepted{0};
    std::atomic<int> rejected{0};
    std::atomic<int> finished{0};
    std::atomic<int> double_terminal{0};
    std::mutex ids_mu;
    std::vector<std::string> all_ids;

    const auto submit_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    const auto poll_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);

    // 4 submitters hammer admission (CAS bound is exercised hardest here)
    std::vector<std::thread> submitters;
    for (int t = 0; t < 4; ++t) {
        submitters.emplace_back([&, t]() {
            while (std::chrono::steady_clock::now() < submit_deadline) {
                const auto out = table.submit(
                    make_req("t" + std::to_string(t), "payload"));
                if (out.status == Table::SubmitStatus::ACCEPTED) {
                    accepted.fetch_add(1);
                    {
                        std::lock_guard<std::mutex> lock(ids_mu);
                        all_ids.push_back(out.job_id);
                    }
                    work.push(out.job_id);
                } else {
                    rejected.fetch_add(1);
                }
            }
        });
    }

    // 2 runners play the server role: take_request, transition, finish
    std::vector<std::thread> runners;
    for (int t = 0; t < 2; ++t) {
        runners.emplace_back([&]() {
            while (const auto id = work.pop()) {
                table.transition_running(*id);
                const auto req = table.take_request(*id);
                if (!req.has_value()) {
                    // evicted before running cannot happen (non-terminal),
                    // but a defensive fail keeps the depth consistent
                    if (table.fail(*id, "request missing")) {
                        finished.fetch_add(1);
                    }
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (table.finish(*id, make_result(static_cast<int>(req->items.size())))) {
                    finished.fetch_add(1);
                } else {
                    double_terminal.fetch_add(1);
                }
            }
        });
    }

    // 3 pollers race snapshot / take_result against the transitions
    std::vector<std::thread> pollers;
    for (int t = 0; t < 3; ++t) {
        pollers.emplace_back([&]() {
            // per-thread PRNG: rand() is not thread-safe (TSAN flags its
            // global state as a data race and exits 66 on the report)
            std::mt19937 rng{static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id()))};
            while (std::chrono::steady_clock::now() < poll_deadline) {
                std::string id;
                {
                    std::lock_guard<std::mutex> lock(ids_mu);
                    if (!all_ids.empty()) {
                        id = all_ids[static_cast<size_t>(rng()) % all_ids.size()];
                    }
                }
                if (id.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                if (const auto snap = table.snapshot(id); snap.has_value()) {
                    if (is_async_terminal(snap->state)) {
                        ASSERT_GE(snap->completed_at_ms, 0);
                    } else {
                        ASSERT_EQ(snap->completed_at_ms, 0);
                    }
                }
                const auto result = table.take_result(id);
                if (result.status == Table::ResultStatus::READY) {
                    ASSERT_EQ(result.value.model_run_status, StatusCode::OK);
                }
            }
        });
    }

    // 2 waiters exercise the condition-variable handoff with tiny timeouts
    std::vector<std::thread> waiters;
    for (int t = 0; t < 2; ++t) {
        waiters.emplace_back([&]() {
            std::mt19937 rng{static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id())) + 1};
            while (std::chrono::steady_clock::now() < poll_deadline) {
                std::string id;
                {
                    std::lock_guard<std::mutex> lock(ids_mu);
                    if (!all_ids.empty()) {
                        id = all_ids[static_cast<size_t>(rng()) % all_ids.size()];
                    }
                }
                if (id.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                if (const auto snap = table.wait(id, AsyncJobState::PENDING, 5);
                    snap.has_value()) {
                    // DONE is the success state and carries no error
                    ASSERT_TRUE(snap->error.empty() ||
                                snap->state != AsyncJobState::DONE);
                }
            }
        });
    }

    for (auto& th : submitters) {
        th.join();
    }
    work.close();
    for (auto& th : runners) {
        th.join();
    }
    for (auto& th : pollers) {
        th.join();
    }
    for (auto& th : waiters) {
        th.join();
    }

    // ---- invariants ----
    EXPECT_GT(accepted.load(), 0);
    EXPECT_EQ(table.queue_depth(), 0);
    EXPECT_EQ(accepted.load(), finished.load());
    EXPECT_EQ(double_terminal.load(), 0);
    std::set<std::string> unique_ids(all_ids.begin(), all_ids.end());
    EXPECT_EQ(unique_ids.size(), static_cast<size_t>(accepted.load()));
    // every non-evicted job must be terminal (evicted ones are gone by design)
    for (const auto& id : unique_ids) {
        if (const auto snap = table.snapshot(id); snap.has_value()) {
            ASSERT_TRUE(is_async_terminal(snap->state)) << "id=" << id;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
