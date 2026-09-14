/************************************************
 * Author: Codex
 * File: batch_collector_unittest.cc
 * Date: 2026-09-13
 ************************************************/

// Focused tests for BatchRequestState::write_slot / notify ordering and a
// lightweight BatchCollector pipeline (inline GoStarter) with a fake worker.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/status_code.h"
#include "models/io/common_input.h"
#include "server/batch_collector.h"
#include "server/inference_task.h"
#include "server/prometheus_metrics.h"
#include "server/sync_request_graph.h"
#include "server/worker_pool.h"

using jinq::common::StatusCode;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::byte_source;
using jinq::server::BatchCollector;
using jinq::server::BatchRequestState;
using jinq::server::PrometheusMetrics;
using jinq::server::WorkerPool;
using jinq::server::assemble_batch_slots;

namespace {

struct FakeOutput {
    int value = 0;
};

struct FakeModel {
    using input_type = base64_input;
    std::atomic<int> batch_calls{0};
    int delay_ms = 0;

    StatusCode run(const base64_input& in, FakeOutput& out) {
        out.value = static_cast<int>(in.input_image_content.size());
        return StatusCode::OK;
    }

    StatusCode run_batch(const std::vector<base64_input>& in, std::vector<FakeOutput>& out,
                         std::vector<StatusCode>& item_status) {
        batch_calls.fetch_add(1);
        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        out.resize(in.size());
        item_status.assign(in.size(), StatusCode::OK);
        for (size_t i = 0; i < in.size(); ++i) {
            out[i].value = static_cast<int>(in[i].input_image_content.size());
        }
        return StatusCode::OK;
    }
};

using FakeWorker = std::unique_ptr<FakeModel>;

byte_source text_item(std::string payload) {
    byte_source item;
    item.origin = byte_source::origin_kind::base64_text;
    item.data = std::move(payload);
    return item;
}

}  // namespace

TEST(batch_collector, write_slot_notifies_once_on_last) {
    auto state = std::make_shared<BatchRequestState<FakeOutput>>();
    state->init(3);
    std::atomic<int> notifies{0};
    state->notify_done = [&]() { notifies.fetch_add(1); };

    BatchRequestState<FakeOutput>::write_slot(state, 0, StatusCode::OK, FakeOutput{1});
    EXPECT_EQ(notifies.load(), 0);
    BatchRequestState<FakeOutput>::write_slot(state, 2, StatusCode::OK, FakeOutput{3});
    EXPECT_EQ(notifies.load(), 0);
    BatchRequestState<FakeOutput>::write_slot(state, 1, StatusCode::OK, FakeOutput{2});
    EXPECT_EQ(notifies.load(), 1);

    EXPECT_TRUE(state->slot_published(0));
    EXPECT_TRUE(state->slot_published(1));
    EXPECT_TRUE(state->slot_published(2));
    EXPECT_EQ(state->outputs[0].value, 1);
    EXPECT_EQ(state->outputs[1].value, 2);
    EXPECT_EQ(state->outputs[2].value, 3);
}


TEST(batch_collector, write_slot_concurrent_same_index_publishes_once) {
    auto state = std::make_shared<BatchRequestState<FakeOutput>>();
    state->init(1);
    std::atomic<int> notifies{0};
    state->notify_done = [&]() { notifies.fetch_add(1); };

    constexpr int kThreads = 8;
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            ready.fetch_add(1);
            while (ready.load() < kThreads) {
            }
            BatchRequestState<FakeOutput>::write_slot(
                state, 0, StatusCode::OK, FakeOutput{100 + t});
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_TRUE(state->slot_published(0));
    EXPECT_EQ(state->completed.load(std::memory_order_acquire), 1u);
    EXPECT_EQ(notifies.load(), 1);
    EXPECT_GE(state->outputs[0].value, 100);
    EXPECT_LT(state->outputs[0].value, 100 + kThreads);
}

TEST(batch_collector, submit_stop_race_all_slots_publish) {
    PrometheusMetrics metrics;
    WorkerPool<FakeWorker> pool;
    pool.adopt(std::make_unique<FakeModel>());
    pool.commit_watermark(1);

    // Empty GoStarter => inline run (unit-test only); focuses the TOCTOU fixup.
    BatchCollector<FakeWorker, FakeOutput> collector(pool, metrics);
    collector.configure(/*max_batch_size=*/4, /*max_batch_delay_ms=*/20,
                        /*worker_wait_timeout_ms=*/200);
    collector.start();

    constexpr int kRequests = 32;
    std::vector<std::shared_ptr<BatchRequestState<FakeOutput>>> states;
    std::vector<std::shared_ptr<std::atomic<int>>> notifies;
    states.reserve(kRequests);
    notifies.reserve(kRequests);
    for (int i = 0; i < kRequests; ++i) {
        auto counter = std::make_shared<std::atomic<int>>(0);
        notifies.push_back(counter);
        auto state = std::make_shared<BatchRequestState<FakeOutput>>();
        state->init(2);
        state->req.items = {text_item("a"), text_item("bb")};
        state->notify_done = [counter]() { counter->fetch_add(1); };
        states.push_back(std::move(state));
    }

    std::atomic<int> go{0};
    std::thread stopper([&]() {
        go.fetch_add(1);
        while (go.load() < 2) {
        }
        collector.stop();
    });
    std::thread submitter([&]() {
        go.fetch_add(1);
        while (go.load() < 2) {
        }
        for (auto& state : states) {
            collector.submit(state);
        }
    });
    stopper.join();
    submitter.join();

    for (int i = 0; i < kRequests; ++i) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (notifies[i]->load() == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(notifies[i]->load(), 1) << "request " << i;
        EXPECT_TRUE(states[i]->slot_published(0)) << "request " << i;
        EXPECT_TRUE(states[i]->slot_published(1)) << "request " << i;
        EXPECT_EQ(states[i]->completed.load(), 2u) << "request " << i;
    }
}

TEST(batch_collector, assemble_batch_slots_timeout_without_acquire) {
    auto state = std::make_shared<BatchRequestState<FakeOutput>>();
    state->init(2);
    BatchRequestState<FakeOutput>::write_slot(state, 0, StatusCode::OK, FakeOutput{7});
    state->find_worker_ms.store(11, std::memory_order_relaxed);
    state->worker_run_ms.store(22, std::memory_order_relaxed);

    auto snap = assemble_batch_slots(*state);
    EXPECT_EQ(snap.model_run_status, StatusCode::DEADLINE_EXCEEDED_PARTIAL);
    EXPECT_TRUE(snap.partial);
    ASSERT_EQ(snap.item_status.size(), 2u);
    EXPECT_EQ(snap.item_status[0], StatusCode::OK);
    EXPECT_EQ(snap.item_outputs[0].value, 7);
    EXPECT_EQ(snap.item_status[1], StatusCode::MODEL_RUN_TIMEOUT);
    EXPECT_EQ(snap.item_outputs[1].value, 0);
    EXPECT_EQ(snap.find_worker_time_consuming, 11.0);
    EXPECT_EQ(snap.worker_run_time_consuming, 22.0);
}

TEST(batch_collector, submit_inline_go_runs_batch_and_notifies) {
    WorkerPool<FakeWorker> pool;
    PrometheusMetrics metrics;
    auto model = std::make_unique<FakeModel>();
    FakeModel* raw = model.get();
    pool.adopt(std::move(model));
    pool.commit_watermark(1);

    // Empty GoStarter => inline run (unit-test only).
    BatchCollector<FakeWorker, FakeOutput> collector(pool, metrics);
    collector.configure(/*max_batch_size=*/4, /*max_batch_delay_ms=*/30,
                        /*worker_wait_timeout_ms=*/500);
    collector.start();

    std::mutex mu;
    std::condition_variable cv;
    bool done = false;

    auto state = std::make_shared<BatchRequestState<FakeOutput>>();
    state->req.items.push_back(text_item("aa"));
    state->req.items.push_back(text_item("bbbb"));
    state->init(2);
    state->notify_done = [&]() {
        std::lock_guard<std::mutex> lock(mu);
        done = true;
        cv.notify_all();
    };

    collector.submit(state);

    {
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&]() { return done; }));
    }

    EXPECT_GE(raw->batch_calls.load(), 1);
    auto snap = assemble_batch_slots(*state);
    EXPECT_EQ(snap.model_run_status, StatusCode::OK);
    ASSERT_EQ(snap.item_status.size(), 2u);
    EXPECT_EQ(snap.item_outputs[0].value, 2);
    EXPECT_EQ(snap.item_outputs[1].value, 4);

    collector.stop();
}


TEST(batch_collector, expired_deadline_skips_worker_run) {
    WorkerPool<FakeWorker> pool;
    PrometheusMetrics metrics;
    auto model = std::make_unique<FakeModel>();
    FakeModel* raw = model.get();
    raw->delay_ms = 50;
    pool.adopt(std::move(model));
    pool.commit_watermark(1);

    BatchCollector<FakeWorker, FakeOutput> collector(pool, metrics);
    collector.configure(/*max_batch_size=*/4, /*max_batch_delay_ms=*/5,
                        /*worker_wait_timeout_ms=*/500);
    collector.start();

    std::mutex mu;
    std::condition_variable cv;
    bool done = false;

    auto state = std::make_shared<BatchRequestState<FakeOutput>>();
    state->req.items.push_back(text_item("aa"));
    state->req.items.push_back(text_item("bb"));
    state->req.deadline =
        std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    state->init(2);
    state->notify_done = [&]() {
        std::lock_guard<std::mutex> lock(mu);
        done = true;
        cv.notify_all();
    };

    collector.submit(state);
    {
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(2), [&]() { return done; }));
    }

    EXPECT_EQ(raw->batch_calls.load(), 0);
    auto snap = assemble_batch_slots(*state);
    EXPECT_EQ(snap.model_run_status, StatusCode::MODEL_RUN_TIMEOUT);
    EXPECT_EQ(snap.item_status[0], StatusCode::MODEL_RUN_TIMEOUT);
    EXPECT_EQ(snap.item_status[1], StatusCode::MODEL_RUN_TIMEOUT);

    collector.stop();
}

TEST(batch_collector, submit_when_stopped_timeouts_all_slots) {
    WorkerPool<FakeWorker> pool;
    PrometheusMetrics metrics;
    BatchCollector<FakeWorker, FakeOutput> collector(pool, metrics);
    collector.configure(4, 10, 100);
    // never start -> not accepting

    std::atomic<int> notifies{0};
    auto state = std::make_shared<BatchRequestState<FakeOutput>>();
    state->req.items.push_back(text_item("x"));
    state->req.items.push_back(text_item("y"));
    state->init(2);
    state->notify_done = [&]() { notifies.fetch_add(1); };

    collector.submit(state);
    EXPECT_EQ(notifies.load(), 1);
    auto snap = assemble_batch_slots(*state);
    EXPECT_EQ(snap.model_run_status, StatusCode::MODEL_RUN_TIMEOUT);
    EXPECT_EQ(snap.item_status[0], StatusCode::MODEL_RUN_TIMEOUT);
    EXPECT_EQ(snap.item_status[1], StatusCode::MODEL_RUN_TIMEOUT);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
