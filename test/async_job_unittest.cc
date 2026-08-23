/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: async_job_unittest.cc
* Date: 26-8-23
************************************************/

// Unit tests for the PRODUCTION async job ledger (server/async_job_table.h).
// The previous version of this file re-implemented the struct and tested the
// copy - which is exactly how the data races survived. These tests compile
// the real component; the stress companion (async_job_stress_test.cc) runs
// the same code under TSAN in CI.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "server/async_job_table.h"

namespace {

using jinq::common::StatusCode;
using jinq::server::AsyncJobState;
using jinq::server::AsyncJobTable;
using jinq::server::is_async_terminal;
using jinq::server::task_request;

struct TrivialOutput {
    int value = 0;
};

using Table = AsyncJobTable<TrivialOutput>;
using Result = jinq::server::go_result<TrivialOutput>;

task_request make_req(const std::string& id, const std::string& payload) {
    task_request req;
    req.task_id = id;
    req.is_valid = true;
    req.parse_status = StatusCode::OK;
    req.payload = payload;
    return req;
}

Result make_result(int value) {
    Result result;
    result.model_run_status = StatusCode::OK;
    result.worker_run_time_consuming = 1.5;
    result.model_output.value = value;
    return result;
}

Table::Config make_cfg(int max_queue, int ttl_ms, int max_completed) {
    Table::Config cfg;
    cfg.max_queue = max_queue;
    cfg.job_ttl_ms = ttl_ms;
    cfg.max_completed = max_completed;
    return cfg;
}

}  // namespace

TEST(async_job_table, terminal_states_ge_done) {
    EXPECT_TRUE(is_async_terminal(AsyncJobState::DONE));
    EXPECT_TRUE(is_async_terminal(AsyncJobState::FAILED));
    EXPECT_TRUE(is_async_terminal(AsyncJobState::TIMEOUT));
    EXPECT_FALSE(is_async_terminal(AsyncJobState::PENDING));
    EXPECT_FALSE(is_async_terminal(AsyncJobState::RUNNING));
}

TEST(async_job_table, submit_assigns_unique_pending_ids) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    std::set<std::string> ids;
    for (int i = 0; i < 8; ++i) {
        const auto out = table.submit(make_req("t" + std::to_string(i), "p"));
        ASSERT_EQ(out.status, Table::SubmitStatus::ACCEPTED);
        ids.insert(out.job_id);
        const auto snap = table.snapshot(out.job_id);
        ASSERT_TRUE(snap.has_value());
        EXPECT_EQ(snap->state, AsyncJobState::PENDING);
        EXPECT_GT(snap->submitted_at_ms, 0);
        EXPECT_EQ(snap->completed_at_ms, 0);
        EXPECT_TRUE(snap->error.empty());
    }
    EXPECT_EQ(ids.size(), 8u);
    EXPECT_EQ(table.queue_depth(), 8);
}

TEST(async_job_table, admission_cas_never_exceeds_max_queue) {
    Table table;
    table.configure(make_cfg(8, 300000, 100));
    // 16 concurrent submissions against 8 slots: no terminal transition ever
    // happens, so the depth never decreases - exactly 8 must be admitted.
    std::atomic<int> accepted{0};
    std::atomic<int> rejected{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 16; ++t) {
        threads.emplace_back([&table, &accepted, &rejected]() {
            const auto out = table.submit(make_req("t", "p"));
            if (out.status == Table::SubmitStatus::ACCEPTED) {
                accepted.fetch_add(1);
            } else {
                rejected.fetch_add(1);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(accepted.load(), 8);
    EXPECT_EQ(rejected.load(), 8);
    EXPECT_EQ(table.queue_depth(), 8);
    EXPECT_LE(table.queue_depth(), 8);
}

TEST(async_job_table, running_jobs_survive_lru_eviction) {
    Table table;
    table.configure(make_cfg(16, 300000, 5));
    std::vector<std::string> ids;
    for (int i = 0; i < 11; ++i) {
        const auto out = table.submit(make_req("t" + std::to_string(i), "p"));
        ASSERT_EQ(out.status, Table::SubmitStatus::ACCEPTED);
        ids.push_back(out.job_id);
    }
    ASSERT_TRUE(table.transition_running(ids[0]));
    for (int i = 1; i < 11; ++i) {
        ASSERT_TRUE(table.finish(ids[i], make_result(i)));
    }
    // one more submit triggers eviction of the oldest terminal jobs
    const auto out = table.submit(make_req("t11", "p"));
    ASSERT_EQ(out.status, Table::SubmitStatus::ACCEPTED);
    EXPECT_TRUE(table.snapshot(ids[0]).has_value());  // running: never evicted
    EXPECT_FALSE(table.snapshot(ids[1]).has_value()); // oldest done: evicted
    EXPECT_FALSE(table.snapshot(ids[5]).has_value());
    EXPECT_TRUE(table.snapshot(ids[6]).has_value());
    EXPECT_TRUE(table.snapshot(ids[10]).has_value());
    EXPECT_TRUE(table.snapshot(out.job_id).has_value());
}

TEST(async_job_table, ttl_evicts_only_completed_after_window) {
    Table table;
    table.configure(make_cfg(16, 80, 100));
    const auto a = table.submit(make_req("a", "p"));
    const auto b = table.submit(make_req("b", "p"));
    ASSERT_EQ(a.status, Table::SubmitStatus::ACCEPTED);
    ASSERT_EQ(b.status, Table::SubmitStatus::ACCEPTED);
    ASSERT_TRUE(table.finish(a.job_id, make_result(1)));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    // a submit triggers the (lazy) retention sweep
    const auto c = table.submit(make_req("c", "p"));
    ASSERT_EQ(c.status, Table::SubmitStatus::ACCEPTED);
    EXPECT_FALSE(table.snapshot(a.job_id).has_value()); // past TTL: evicted
    EXPECT_TRUE(table.snapshot(b.job_id).has_value());  // pending: kept
    EXPECT_TRUE(table.snapshot(c.job_id).has_value());
}

TEST(async_job_table, wait_wakes_on_terminal_within_bound) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    const auto out = table.submit(make_req("t", "p"));
    ASSERT_EQ(out.status, Table::SubmitStatus::ACCEPTED);
    std::thread finisher([&table, &out]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ASSERT_TRUE(table.transition_running(out.job_id));
        ASSERT_TRUE(table.finish(out.job_id, make_result(7)));
    });
    const auto t0 = std::chrono::steady_clock::now();
    const auto snap = table.wait(out.job_id, AsyncJobState::PENDING, 5000);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count();
    finisher.join();
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->state, AsyncJobState::DONE);
    // a correct CV handoff returns at the transition, not at the 5s timeout
    EXPECT_LT(elapsed, 1500);
}

TEST(async_job_table, wait_on_terminal_returns_immediately) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    const auto out = table.submit(make_req("t", "p"));
    ASSERT_TRUE(table.finish(out.job_id, make_result(1)));
    const auto t0 = std::chrono::steady_clock::now();
    const auto snap = table.wait(out.job_id, AsyncJobState::RUNNING, 5000);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count();
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->state, AsyncJobState::DONE);
    EXPECT_LT(elapsed, 500);
}

TEST(async_job_table, result_readable_only_when_done_and_repeatable) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    const auto out = table.submit(make_req("req-1", "p"));
    ASSERT_TRUE(table.transition_running(out.job_id));

    auto before = table.take_result(out.job_id);
    EXPECT_EQ(before.status, Table::ResultStatus::NOT_DONE);
    EXPECT_EQ(before.state, AsyncJobState::RUNNING);

    ASSERT_TRUE(table.finish(out.job_id, make_result(42)));
    for (int i = 0; i < 3; ++i) {
        const auto got = table.take_result(out.job_id);
        ASSERT_EQ(got.status, Table::ResultStatus::READY);
        EXPECT_EQ(got.value.model_output.value, 42);
        EXPECT_EQ(got.value.model_run_status, StatusCode::OK);
        EXPECT_DOUBLE_EQ(got.value.worker_run_time_consuming, 1.5);
        EXPECT_EQ(got.task_id, "req-1");  // request-id echo survives the payload move
    }
}

TEST(async_job_table, fail_and_timeout_record_error_and_release_depth) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    const auto a = table.submit(make_req("a", "p"));
    const auto b = table.submit(make_req("b", "p"));
    ASSERT_TRUE(table.fail(a.job_id, "model run session failed"));
    ASSERT_TRUE(table.timeout(b.job_id, "worker wait timeout"));

    const auto sa = table.snapshot(a.job_id);
    ASSERT_TRUE(sa.has_value());
    EXPECT_EQ(sa->state, AsyncJobState::FAILED);
    EXPECT_EQ(sa->error, "model run session failed");
    EXPECT_GT(sa->completed_at_ms, 0);

    const auto sb = table.snapshot(b.job_id);
    ASSERT_TRUE(sb.has_value());
    EXPECT_EQ(sb->state, AsyncJobState::TIMEOUT);
    EXPECT_EQ(sb->error, "worker wait timeout");

    EXPECT_EQ(table.queue_depth(), 0);
}

TEST(async_job_table, take_request_moves_payload_once_and_keeps_task_id) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    const auto out = table.submit(make_req("req-9", "payload-data"));
    const auto first = table.take_request(out.job_id);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->task_id, "req-9");
    EXPECT_EQ(first->payload, "payload-data");
    const auto second = table.take_request(out.job_id);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->task_id, "req-9");
    EXPECT_TRUE(second->payload.empty());  // moved out once
    ASSERT_TRUE(table.finish(out.job_id, make_result(1)));
    const auto got = table.take_result(out.job_id);
    ASSERT_EQ(got.status, Table::ResultStatus::READY);
    EXPECT_EQ(got.task_id, "req-9");
}

TEST(async_job_table, terminal_transitions_are_exactly_once) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    const auto out = table.submit(make_req("t", "p"));
    EXPECT_TRUE(table.transition_running(out.job_id));
    EXPECT_FALSE(table.transition_running(out.job_id));  // already running
    EXPECT_TRUE(table.finish(out.job_id, make_result(1)));
    EXPECT_FALSE(table.finish(out.job_id, make_result(2)));
    EXPECT_FALSE(table.fail(out.job_id, "late"));
    EXPECT_FALSE(table.timeout(out.job_id, "late"));
    const auto snap = table.snapshot(out.job_id);
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->state, AsyncJobState::DONE);
    const auto got = table.take_result(out.job_id);
    ASSERT_EQ(got.status, Table::ResultStatus::READY);
    EXPECT_EQ(got.value.model_output.value, 1);  // first result wins
    EXPECT_EQ(table.queue_depth(), 0);           // decremented exactly once
}

TEST(async_job_table, missing_job_operations_return_not_found) {
    Table table;
    table.configure(make_cfg(16, 300000, 100));
    EXPECT_FALSE(table.snapshot("nope").has_value());
    EXPECT_FALSE(table.wait("nope", AsyncJobState::PENDING, 1).has_value());
    EXPECT_FALSE(table.take_request("nope").has_value());
    EXPECT_EQ(table.take_result("nope").status, Table::ResultStatus::NOT_FOUND);
    EXPECT_FALSE(table.transition_running("nope"));
    EXPECT_FALSE(table.finish("nope", make_result(1)));
    EXPECT_FALSE(table.fail("nope", "x"));
    EXPECT_FALSE(table.timeout("nope", "x"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
