/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: rate_limit_gcra_bench.cc
* Date: 26-9-15
************************************************/

// Manual performance baseline for the GCRA kernel (NOT a CI test; build with
//   cmake --build <build> --target rate_limit_gcra_bench
// ). PR-1 acceptance reference: mixed-key admit p50 < 1us, p99 < 10us on one
// core. Numbers are machine-dependent — record them in the PR description.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "control/rate_limit/gcra.h"

namespace {

using mortred::control::ratelimit::GcraPolicy;
using mortred::control::ratelimit::ShardedIpLimiter;
using mortred::control::ratelimit::SubjectKey;

SubjectKey key4(uint32_t ip) {
    SubjectKey k;
    k.family = 4;
    k.prefix[0] = static_cast<uint8_t>(ip >> 24);
    k.prefix[1] = static_cast<uint8_t>(ip >> 16);
    k.prefix[2] = static_cast<uint8_t>(ip >> 8);
    k.prefix[3] = static_cast<uint8_t>(ip);
    return k;
}

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double percentile_ns(std::vector<int64_t> samples, double p) {
    std::sort(samples.begin(), samples.end());
    const size_t idx = std::min(samples.size() - 1,
                                static_cast<size_t>(p * samples.size()));
    return static_cast<double>(samples[idx]);
}

void report(const char* name, const std::vector<int64_t>& samples, size_t ops) {
    std::printf("%-22s ops=%zu  p50=%.0fns  p99=%.0fns  max=%.0fns\n", name, ops,
                percentile_ns(samples, 0.50), percentile_ns(samples, 0.99),
                percentile_ns(samples, 1.0));
}

}  // namespace

int main() {
    // hot policy: interval 10ms, burst 50 -> always contested under hammering
    const GcraPolicy policy{100, 50};
    ShardedIpLimiter limiter(policy, 262'144);

    // ---- single-threaded: mixed 10k keys, round-robin, wall-clock ts ----
    {
        constexpr size_t kKeys = 10'000;
        constexpr size_t kOps = 1'000'000;
        std::vector<SubjectKey> keys;
        keys.reserve(kKeys);
        for (uint32_t i = 0; i < kKeys; ++i) {
            keys.push_back(key4(0x0a000000 + i));
        }
        std::vector<int64_t> samples;
        samples.reserve(kOps);
        uint64_t rejected = 0;
        for (size_t op = 0; op < kOps; ++op) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!limiter.admit(keys[op % kKeys], now_ms()).allowed) {
                ++rejected;
            }
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        std::printf("single-thread mixed 10k keys: rejected=%llu/%zu\n",
                    static_cast<unsigned long long>(rejected), kOps);
        report("  per-admit", samples, kOps);
    }

    // ---- single-threaded: one hot key (worst case: same shard+cell) ----
    {
        constexpr size_t kOps = 1'000'000;
        const SubjectKey hot = key4(0x0a0000ff);
        std::vector<int64_t> samples;
        samples.reserve(kOps);
        for (size_t op = 0; op < kOps; ++op) {
            const auto t0 = std::chrono::steady_clock::now();
            limiter.admit(hot, now_ms());
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
        report("  hot-key", samples, kOps);
    }

    // ---- multi-threaded aggregate ----
    {
        constexpr int kThreads = 4;
        constexpr size_t kPerThread = 1'000'000;
        std::atomic<uint64_t> done{0};
        std::vector<std::thread> threads;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                const SubjectKey k = key4(0xc0a80000 + t);
                for (size_t i = 0; i < kPerThread; ++i) {
                    limiter.admit(k, now_ms());
                    done.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
        const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        std::printf("multi-thread 4x1M same-key-per-thread: %.0f ops/s aggregate\n",
                    static_cast<double>(done.load()) * 1e9 / static_cast<double>(dt));
    }

    const auto stats = limiter.stats();
    std::printf("stats: tracked=%zu rejects=%llu evictions=%llu saturations=%llu\n",
                stats.tracked, static_cast<unsigned long long>(stats.rejects),
                static_cast<unsigned long long>(stats.evictions),
                static_cast<unsigned long long>(stats.saturations));
    return 0;
}
