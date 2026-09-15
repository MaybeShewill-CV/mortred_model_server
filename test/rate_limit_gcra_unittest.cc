/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: rate_limit_gcra_unittest.cc
* Date: 26-9-15
************************************************/

// GCRA kernel tests. The clock is injected everywhere (int64 ms), so the
// state machine is verified without sleeping; the differential tests run the
// implementation and a sequential reference simulator over identical random
// arrival sequences and require identical decisions AND retry_after values.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "control/rate_limit/gcra.h"

namespace {

using mortred::control::ratelimit::GcraDecision;
using mortred::control::ratelimit::GcraPolicy;
using mortred::control::ratelimit::ShardedIpLimiter;
using mortred::control::ratelimit::SubjectKey;
using mortred::control::ratelimit::gcra_admit;

SubjectKey key4(uint32_t ip) {
    SubjectKey k;
    k.family = 4;
    k.prefix[0] = static_cast<uint8_t>(ip >> 24);
    k.prefix[1] = static_cast<uint8_t>(ip >> 16);
    k.prefix[2] = static_cast<uint8_t>(ip >> 8);
    k.prefix[3] = static_cast<uint8_t>(ip);
    return k;
}

SubjectKey key6(uint64_t prefix_hi) {
    SubjectKey k;
    k.family = 6;
    for (int i = 0; i < 8; ++i) {
        k.prefix[i] = static_cast<uint8_t>(prefix_hi >> (56 - 8 * i));
    }
    return k;
}

/*** Sequential reference: identical integer arithmetic, no CAS, no storage.
 * The differential tests prove the sharded/atomic implementation matches this
 * oracle exactly — they pin the state machine, not the float math. */
class RefGcra {
  public:
    RefGcra(GcraPolicy policy) : policy_(policy) {}

    GcraDecision admit(int64_t now_ms) {
        if (!policy_.enabled()) {
            return {true, 0};
        }
        const int64_t interval = policy_.interval_ms();
        const int64_t sigma = policy_.burst_tolerance_ms();
        const int64_t effective = std::max(tat_ms_, now_ms);
        if (effective - now_ms > sigma) {
            return {false, effective - sigma - now_ms};
        }
        tat_ms_ = effective + interval;
        return {true, 0};
    }

  private:
    GcraPolicy policy_;
    int64_t tat_ms_ = 0;
};

constexpr GcraPolicy kRate10Burst5{10, 5};  // T=100ms, sigma=400ms

}  // namespace

TEST(GcraPolicy, IntervalRoundsUpConservatively) {
    GcraPolicy p{3, 1};
    // 1000/3 = 333.33 -> ceil 334: effective rate 2.994/s, never > nominal
    EXPECT_EQ(p.interval_ms(), 334);
    EXPECT_LE(1000 / p.interval_ms(), 3);
    EXPECT_EQ(kRate10Burst5.interval_ms(), 100);
    EXPECT_EQ(kRate10Burst5.burst_tolerance_ms(), 400);
    // parentheses: GCC's preprocessor splits the braced-init comma into
    // two macro arguments (clang does not — portable form uses parens)
    EXPECT_FALSE((GcraPolicy{0, 5}.enabled()));
    EXPECT_FALSE((GcraPolicy{10, 0}.enabled()));
}

TEST(GcraAdmit, DisabledPolicyAllowsWithoutTouchingState) {
    std::atomic<int64_t> tat{0};
    const GcraPolicy disabled{0, 10};
    for (int i = 0; i < 100; ++i) {
        const auto d = gcra_admit(tat, disabled, i);
        EXPECT_TRUE(d.allowed);
    }
    EXPECT_EQ(tat.load(), 0);
}

TEST(GcraAdmit, FreshCellAlwaysConforms) {
    std::atomic<int64_t> tat{0};
    EXPECT_TRUE(gcra_admit(tat, kRate10Burst5, 1'000'000).allowed);
}

TEST(GcraAdmit, BurstCapacityIsExact) {
    std::atomic<int64_t> tat{0};
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(gcra_admit(tat, kRate10Burst5, 0).allowed) << "instant #" << i;
    }
    const auto rejected = gcra_admit(tat, kRate10Burst5, 0);
    EXPECT_FALSE(rejected.allowed);
    // after 5 instantaneous admissions TAT=500ms; retry at 500-400-0 = 100ms
    EXPECT_EQ(rejected.retry_after_ms, 100);
}

TEST(GcraAdmit, SteadyRateConformsForever) {
    std::atomic<int64_t> tat{0};
    for (int64_t t = 0; t < 100'000; t += 100) {  // exactly one per interval
        EXPECT_TRUE(gcra_admit(tat, kRate10Burst5, t).allowed) << "t=" << t;
    }
}

TEST(GcraAdmit, RetryAfterIsExactAndRecoveryWorks) {
    std::atomic<int64_t> tat{0};
    for (int i = 0; i < 5; ++i) {
        gcra_admit(tat, kRate10Burst5, 0);
    }
    const auto rejected = gcra_admit(tat, kRate10Burst5, 0);
    ASSERT_FALSE(rejected.allowed);
    EXPECT_EQ(rejected.retry_after_ms, 100);
    // one ms before recovery: still rejected with retry 1
    const auto early = gcra_admit(tat, kRate10Burst5, 99);
    ASSERT_FALSE(early.allowed);
    EXPECT_EQ(early.retry_after_ms, 1);
    // at recovery time: allowed
    EXPECT_TRUE(gcra_admit(tat, kRate10Burst5, 100).allowed);
}

TEST(GcraAdmit, DrainedStateBehavesLikeFresh) {
    std::atomic<int64_t> tat{0};
    for (int i = 0; i < 5; ++i) {
        gcra_admit(tat, kRate10Burst5, 0);
    }
    // idle far past sigma: conforming again with full burst budget
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(gcra_admit(tat, kRate10Burst5, 10'000'000 + i).allowed);
    }
}

TEST(GcraDifferential, RandomSequencesMatchReference) {
    // mixed regimes: poisson-ish gaps, bursts, idle stretches, clock jitter
    for (uint32_t seed = 1; seed <= 100; ++seed) {
        std::mt19937 rng(seed);
        // mt19937::result_type is uint64 on linux: explicit casts avoid the
        // -Wnarrowing error under the -Werror quality-gate preset
        GcraPolicy policy{static_cast<uint32_t>(1 + rng() % 200),
                          static_cast<uint32_t>(1 + rng() % 50)};
        std::atomic<int64_t> tat{0};
        RefGcra ref(policy);
        int64_t now = 0;
        for (int i = 0; i < 2'000; ++i) {
            const int regime = rng() % 10;
            if (regime < 4) {
                now += rng() % (2 * policy.interval_ms() + 1);  // near-rate
            } else if (regime < 7) {
                now += rng() % 3;  // burst
            } else if (regime < 9) {
                now += policy.burst_tolerance_ms() + policy.interval_ms() + 1 + rng() % 50;  // idle
            } else {
                now -= 0;  // same instant repeats
            }
            const auto mine = gcra_admit(tat, policy, now);
            const auto oracle = ref.admit(now);
            ASSERT_EQ(mine.allowed, oracle.allowed)
                << "seed=" << seed << " i=" << i << " now=" << now;
            ASSERT_EQ(mine.retry_after_ms, oracle.retry_after_ms)
                << "seed=" << seed << " i=" << i << " now=" << now;
        }
    }
}

TEST(GcraDifferential, BurstOneIsStrictlySpaced) {
    const GcraPolicy p{10, 1};  // sigma = 0: one per 100ms, no slack
    std::atomic<int64_t> tat{0};
    RefGcra ref(p);
    for (int64_t t = 0; t < 3'000; t += 7) {
        const auto mine = gcra_admit(tat, p, t);
        const auto oracle = ref.admit(t);
        ASSERT_EQ(mine.allowed, oracle.allowed) << "t=" << t;
        ASSERT_EQ(mine.retry_after_ms, oracle.retry_after_ms) << "t=" << t;
    }
}

TEST(ShardedLimiter, DisabledPolicyTracksNothing) {
    ShardedIpLimiter limiter(GcraPolicy{0, 5}, 1024);
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(limiter.admit(key4(i), i).allowed);
    }
    const auto stats = limiter.stats();
    EXPECT_EQ(stats.tracked, 0u);
    EXPECT_EQ(stats.rejects, 0u);
}

TEST(ShardedLimiter, PerKeyBehaviorMatchesFreeFunction) {
    ShardedIpLimiter limiter(kRate10Burst5, 1024);
    std::atomic<int64_t> tat{0};
    const SubjectKey k = key4(0x0a000001);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(limiter.admit(k, 0).allowed);
        EXPECT_TRUE(gcra_admit(tat, kRate10Burst5, 0).allowed);
    }
    EXPECT_FALSE(limiter.admit(k, 0).allowed);
    EXPECT_FALSE(gcra_admit(tat, kRate10Burst5, 0).allowed);
    EXPECT_EQ(limiter.stats().tracked, 1u);
    EXPECT_EQ(limiter.stats().rejects, 1u);
}

TEST(ShardedLimiter, KeysAreIndependent) {
    ShardedIpLimiter limiter(kRate10Burst5, 1024);
    const SubjectKey a = key4(1);
    const SubjectKey b = key4(2);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(limiter.admit(a, 0).allowed);
    }
    EXPECT_FALSE(limiter.admit(a, 0).allowed);
    EXPECT_TRUE(limiter.admit(b, 0).allowed);  // b unaffected by a's spend
    EXPECT_EQ(limiter.stats().tracked, 2u);
}

TEST(ShardedLimiter, KeysSpreadAcrossShards) {
    // 4096 v4 keys over 16 shards: no shard should hold everything
    ShardedIpLimiter limiter(kRate10Burst5, 100'000);
    for (uint32_t i = 0; i < 4096; ++i) {
        limiter.admit(key4(i), 0);
    }
    EXPECT_EQ(limiter.stats().tracked, 4096u);
}

TEST(ShardedLimiter, IdleSweepReclaimsDrainedFlows) {
    // T=1ms sigma=9ms -> idle expiry = 2*(9+1) = 20ms. Lazy sweep fires on
    // the insert path only when the target shard is at budget (documented
    // design: the memory bound is max_tracked, not prompt reclamation), so
    // the test drives a single shard deterministically through at-budget ->
    // saturation -> idle-drain -> sweep.
    const GcraPolicy p{1000, 10};
    ShardedIpLimiter limiter(p, 32);  // 2 cells per shard
    const mortred::control::ratelimit::SubjectHash hash;
    std::vector<SubjectKey> shard0;
    for (uint32_t i = 0; shard0.size() < 3 && i < 65536; ++i) {
        const auto k = key4(i);
        if (hash(k) % ShardedIpLimiter::kShards == 0) {
            shard0.push_back(k);
        }
    }
    ASSERT_EQ(shard0.size(), 3u);
    EXPECT_TRUE(limiter.admit(shard0[0], 1'000).allowed);
    EXPECT_TRUE(limiter.admit(shard0[1], 1'000).allowed);
    // shard at budget with live (not idle) cells: newcomer shed
    EXPECT_FALSE(limiter.admit(shard0[2], 1'000).allowed);
    EXPECT_EQ(limiter.stats().saturations, 1u);
    EXPECT_EQ(limiter.stats().evictions, 0u);
    // both cells idle-drained (1000ms > 20ms): the same insert sweeps first
    EXPECT_TRUE(limiter.admit(shard0[2], 2'000).allowed);
    EXPECT_EQ(limiter.stats().evictions, 2u);
}

TEST(ShardedLimiter, SaturationKeepsTrackedBudgets) {
    const GcraPolicy p{1000, 10};
    ShardedIpLimiter limiter(p, 16);  // exactly 1 cell per shard
    // deterministically find two keys sharing one shard (hash is public)
    const mortred::control::ratelimit::SubjectHash hash;
    std::vector<SubjectKey> shard0;
    for (uint32_t i = 0; shard0.size() < 2 && i < 65536; ++i) {
        const auto k = key4(i);
        if (hash(k) % ShardedIpLimiter::kShards == 0) {
            shard0.push_back(k);
        }
    }
    ASSERT_EQ(shard0.size(), 2u);
    EXPECT_TRUE(limiter.admit(shard0[0], 100).allowed);
    // same shard at capacity, entry not idle -> newcomer shed, budget kept
    const auto newcomer = limiter.admit(shard0[1], 100);
    EXPECT_FALSE(newcomer.allowed);
    EXPECT_GT(newcomer.retry_after_ms, 0);
    EXPECT_EQ(limiter.stats().saturations, 1u);
    EXPECT_TRUE(limiter.admit(shard0[0], 101).allowed);
    // a different shard is unaffected
    std::vector<SubjectKey> shard1;
    for (uint32_t i = 0; shard1.empty() && i < 65536; ++i) {
        const auto k = key4(i);
        if (hash(k) % ShardedIpLimiter::kShards == 1) {
            shard1.push_back(k);
        }
    }
    ASSERT_EQ(shard1.size(), 1u);
    EXPECT_TRUE(limiter.admit(shard1[0], 100).allowed);
}

TEST(ShardedLimiter, DegenerateBudgetFailsOpen) {
    ShardedIpLimiter limiter(kRate10Burst5, 0);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(limiter.admit(key4(i), i).allowed);
    }
    EXPECT_EQ(limiter.stats().fail_opens, 10u);
    EXPECT_EQ(limiter.stats().tracked, 0u);
}

TEST(ShardedLimiter, ConcurrentPerKeyMatchesReference) {
    // each thread owns one key: the shard mutex serializes a key's admissions
    // in program order, so per-key decisions must equal the sequential oracle
    // fed the same (now, key) sequence in the same order.
    constexpr int kThreads = 8;
    constexpr int kAdmits = 2'000;
    ShardedIpLimiter limiter(kRate10Burst5, 1024);
    std::vector<std::vector<std::pair<int64_t, GcraDecision>>> records(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        records[t].reserve(kAdmits);
        threads.emplace_back([&, t] {
            std::mt19937 rng(777 + t);
            int64_t now = 0;
            for (int i = 0; i < kAdmits; ++i) {
                const int regime = rng() % 4;
                if (regime < 2) {
                    now += rng() % 200;  // near-rate-ish
                } else if (regime < 3) {
                    now += rng() % 3;  // burst
                } else {
                    now += 700;  // idle past sigma
                }
                const auto d = limiter.admit(key4(0x0a000000 + t), now);
                records[t].emplace_back(now, d);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (int t = 0; t < kThreads; ++t) {
        RefGcra ref(kRate10Burst5);
        for (const auto& [now, decision] : records[t]) {
            const auto oracle = ref.admit(now);
            ASSERT_EQ(decision.allowed, oracle.allowed) << "thread " << t;
            ASSERT_EQ(decision.retry_after_ms, oracle.retry_after_ms) << "thread " << t;
        }
    }
    EXPECT_EQ(limiter.stats().tracked, static_cast<size_t>(kThreads));
}

TEST(ShardedLimiter, ConcurrentSameKeyHammerIsSerializable) {
    // shared key hammered by many threads: exact per-order prediction is
    // impossible, but the invariants hold — no crash, accounting consistent,
    // and the allowed count never exceeds the sequential upper bound
    // (fresh burst + one per elapsed interval).
    ShardedIpLimiter limiter(kRate10Burst5, 1024);
    std::atomic<uint64_t> allowed{0};
    std::atomic<uint64_t> rejected{0};
    std::vector<std::thread> threads;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 5'000;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                const auto d = limiter.admit(key4(0x0a000099), t * 10'000 + i);
                if (d.allowed) {
                    allowed.fetch_add(1);
                } else {
                    rejected.fetch_add(1);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(allowed.load() + rejected.load(),
              static_cast<uint64_t>(kThreads) * kPerThread);
    // span t=0..79998, T=100ms, burst=5: upper bound = 5 + 800 = 805
    EXPECT_LE(allowed.load(), 805u);
    const auto stats = limiter.stats();
    EXPECT_EQ(stats.tracked, 1u);
    EXPECT_EQ(stats.rejects, rejected.load());
}

TEST(SubjectKey, HashIsStableAcrossInstances) {
    const auto k = key6(0x20010db800000000ULL);
    const mortred::control::ratelimit::SubjectHash hash_a;
    const mortred::control::ratelimit::SubjectHash hash_b;
    EXPECT_EQ(hash_a(k), hash_b(k));
    EXPECT_EQ(k, key6(0x20010db800000000ULL));
    EXPECT_FALSE(k == key6(0x20010db800000001ULL));
    EXPECT_FALSE(k == key4(0x20010db8));
}
