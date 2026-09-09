/************************************************
 * Author: Codex
 * File: worker_pool_unittest.cc
 * Date: 2026-09-09
 ************************************************/

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "server/worker_pool.h"

using jinq::server::StuckWorkerAction;
using jinq::server::WorkerPool;

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
