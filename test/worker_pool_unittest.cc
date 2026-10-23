/************************************************
 * Author: Codex
 * File: worker_pool_unittest.cc
 * Date: 2026-09-09
 ************************************************/

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "glog/logging.h"
#include "server/worker_pool.h"

using mortred::server::StuckWorkerAction;
using mortred::server::WorkerPool;

namespace {

std::atomic<int64_t> g_now_ms{0};

int64_t fake_now_ms() {
    return g_now_ms.load();
}

using IntWorker = std::unique_ptr<int>;

}  // namespace

TEST(worker_pool, checkout_timeout_then_success_resets_streak) {
    g_now_ms.store(0);
    WorkerPool<IntWorker> pool(&fake_now_ms);
    pool.configure_stuck(StuckWorkerAction::LOG, 2, 100);
    pool.commit_watermark(1);

    IntWorker worker;
    const auto miss = pool.checkout(worker, std::chrono::milliseconds(20));
    EXPECT_FALSE(miss.ok);

    pool.adopt(std::make_unique<int>(7));
    const auto hit = pool.checkout(worker, std::chrono::milliseconds(50));
    ASSERT_TRUE(hit.ok);
    ASSERT_NE(worker, nullptr);
    EXPECT_EQ(*worker, 7);
    pool.checkin(std::move(worker));
    EXPECT_EQ(pool.available_approx(), 1u);
}

TEST(worker_pool, stuck_log_uses_injected_clock_not_async_timeout) {
    g_now_ms.store(0);
    WorkerPool<IntWorker> pool(&fake_now_ms);
    // span threshold is K * model_run_timeout (100), never async_timeout
    pool.configure_stuck(StuckWorkerAction::LOG, 2, 100);

    IntWorker worker;
    g_now_ms.store(0);
    EXPECT_FALSE(pool.checkout(worker, std::chrono::milliseconds(15)).ok);
    g_now_ms.store(200);
    EXPECT_FALSE(pool.checkout(worker, std::chrono::milliseconds(15)).ok);
}

TEST(worker_pool, unbounded_wait_and_ewma) {
    WorkerPool<IntWorker> pool;
    pool.seed_ewma(500);
    pool.observe_run_ms(100);
    EXPECT_EQ(pool.ewma_ms(), 420);

    std::atomic<bool> started{false};
    std::thread producer([&] {
        while (!started.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        pool.adopt(std::make_unique<int>(9));
    });
    IntWorker worker;
    started.store(true);
    const auto ck = pool.checkout(worker, std::chrono::milliseconds(-1));
    EXPECT_TRUE(ck.ok);
    ASSERT_NE(worker, nullptr);
    EXPECT_EQ(*worker, 9);
    producer.join();
}

TEST(worker_pool, drain_timeout_returns_with_remaining_log) {
    WorkerPool<IntWorker> pool;
    pool.adopt(std::make_unique<int>(1));
    pool.commit_watermark(1);
    IntWorker held;
    ASSERT_TRUE(pool.checkout(held, std::chrono::milliseconds(50)).ok);

    FLAGS_logtostderr = true;
    testing::internal::CaptureStderr();
    const auto start = std::chrono::steady_clock::now();
    const bool drained = pool.drain(start + std::chrono::milliseconds(200));
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(drained);
    EXPECT_LT(elapsed, std::chrono::seconds(2));
    EXPECT_NE(log.find("drain timeout"), std::string::npos) << log;
    EXPECT_NE(log.find("remaining=1"), std::string::npos) << log;
    pool.checkin(std::move(held));
    EXPECT_TRUE(pool.drain(std::chrono::steady_clock::now() + std::chrono::milliseconds(200)));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
    return RUN_ALL_TESTS();
}
