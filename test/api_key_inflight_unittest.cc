// Gateway wires try_acquire_inflight/release_inflight around upstream forward (R5 P1-3).
#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "control/api_key_manager.h"

using mortred::control::ApiKey;
using mortred::control::ApiKeyManager;

TEST(ApiKeyInflight, unlimited_when_zero) {
    ApiKey k;
    k.max_inflight = 0;
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
}

TEST(ApiKeyInflight, enforces_budget) {
    ApiKey k;
    k.max_inflight = 2;
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_FALSE(ApiKeyManager::try_acquire_inflight(&k));
    ApiKeyManager::release_inflight(&k);
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
}

TEST(ApiKeyInflight, tracks_when_worker_budget_set) {
    ApiKey k;
    k.max_inflight = 0;
    k.worker_budget = 2;
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_EQ(k.inflight.load(), 1);
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_EQ(k.inflight.load(), 2);
    ApiKeyManager::release_inflight(&k);
    EXPECT_EQ(k.inflight.load(), 1);
    ApiKeyManager::release_inflight(&k);
    EXPECT_EQ(k.inflight.load(), 0);
}

// Round 9: worker_budget enforced inside CAS (N succeed, N+1 fails; no sleep).
TEST(ApiKeyInflight, enforces_worker_budget) {
    ApiKey k;
    k.max_inflight = 0;
    k.worker_budget = 2;
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_FALSE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_EQ(k.inflight.load(), 2);
    EXPECT_GE(k.reject_inflight.load(), 1u);
    ApiKeyManager::release_inflight(&k);
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_EQ(k.inflight.load(), 2);
}

TEST(ApiKeyInflight, worker_budget_tighter_than_max_inflight) {
    ApiKey k;
    k.max_inflight = 8;
    k.worker_budget = 1;
    EXPECT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_FALSE(ApiKeyManager::try_acquire_inflight(&k));
    EXPECT_EQ(k.inflight.load(), 1);
}

// Optional two-thread stress: prefill to budget, concurrent excess must fail; peak never exceeds N.
TEST(ApiKeyInflight, worker_budget_two_thread_stress) {
    ApiKey k;
    k.max_inflight = 0;
    k.worker_budget = 4;
    for (int i = 0; i < k.worker_budget; ++i) {
        ASSERT_TRUE(ApiKeyManager::try_acquire_inflight(&k));
    }
    EXPECT_EQ(k.inflight.load(), 4);

    constexpr int kThreads = 2;
    constexpr int kAttemptsPerThread = 100;
    std::atomic<int> failures{0};
    std::atomic<int> unexpected_ok{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < kAttemptsPerThread; ++i) {
                if (ApiKeyManager::try_acquire_inflight(&k)) {
                    unexpected_ok.fetch_add(1, std::memory_order_relaxed);
                    ApiKeyManager::release_inflight(&k);
                } else {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(unexpected_ok.load(), 0);
    EXPECT_EQ(failures.load(), kThreads * kAttemptsPerThread);
    EXPECT_EQ(k.inflight.load(), 4);
    EXPECT_GE(k.reject_inflight.load(), static_cast<uint64_t>(kThreads * kAttemptsPerThread));

    for (int i = 0; i < k.worker_budget; ++i) {
        ApiKeyManager::release_inflight(&k);
    }
    EXPECT_EQ(k.inflight.load(), 0);

    // Free-for-all: peak inflight observed after successful CAS never exceeds budget.
    std::atomic<int> peak{0};
    threads.clear();
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 500; ++i) {
                if (ApiKeyManager::try_acquire_inflight(&k)) {
                    const int cur = k.inflight.load(std::memory_order_relaxed);
                    int prev = peak.load(std::memory_order_relaxed);
                    while (cur > prev && !peak.compare_exchange_weak(prev, cur)) {
                    }
                    ApiKeyManager::release_inflight(&k);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(k.inflight.load(), 0);
    EXPECT_LE(peak.load(), k.worker_budget);
    EXPECT_GT(peak.load(), 0);
}
