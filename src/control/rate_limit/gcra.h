/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gcra.h
* Date: 26-9-15
************************************************/

// GCRA (Generic Cell Rate Algorithm / TAT) per-source metering kernel for the
// gateway edge rate limiter. Pure algorithm + sharded in-process store:
// workflow-free, toml-free, metrics-free — directly unit-testable in the
// tests-only CI. The gateway wiring lives in gateway_app.cpp (PR-3b); source
// derivation (trusted proxy / Forwarded / XFF / prefix folding) is subject.h
// (PR-2) and produces the SubjectKey consumed here.
//
// GCRA semantics (per flow, state = one int64 "theoretical arrival time"):
//
//   T  = ceil(1000 / rate_per_sec) ms      emission interval (conservative)
//   σ  = (burst - 1) * T                   burst tolerance, chosen so that
//                                         `burst` is the EXACT number of
//                                         admissions possible in ~zero time
//   conforming  ⇔  TAT - now ≤ σ           then TAT = max(TAT, now) + T
//   rejecting   →  retry_after = TAT - σ - now   (exact, ≥ 1 ms by integrality)
//
// Sustained rate ≤ rate_per_sec conforms forever; above it, conforming until
// the bucket is spent, then periodic one-per-interval drips. State that has
// fully drained (TAT far in the past) behaves like a fresh flow: a first
// arrival always conforms.
//
// Capacity & failure policy of ShardedIpLimiter (attack-shaped, memory-bounded):
//   - kShards mutex-sharded maps; every admit touches exactly one shard.
//   - Lazy sweep: inserts into a shard at budget reclaim entries idle for
//     more than idle_expiry = 2*(σ+T) (a fully drained, silent flow — losing
//     its state only forgives debt it does not have).
//   - max_tracked is a hard bound. At capacity the limiter REJECTS previously
//     unseen sources (fail-closed for untracked) while tracked sources keep
//     their budgets: a table-filling new-source flood must not degrade
//     protection for everyone already tracked.
//   - Allocation failure fails OPEN (rate limiting protects availability, it
//     must not become a new outage) and is counted.
//   - A degenerate configuration (max_tracked == 0) fails open for every new
//     source instead of rejecting everything; counted separately.

#ifndef MORTRED_CONTROL_RATE_LIMIT_GCRA_H
#define MORTRED_CONTROL_RATE_LIMIT_GCRA_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

namespace mortred {
namespace control {
namespace ratelimit {

/*** Sustained-rate + burst policy of one metering class.
 * rate_per_sec == 0 disables metering (every admission conforms, state is
 * not touched). burst >= 1 required for an enabled policy; burst == 1 means
 * strictly one admission per interval with no instantaneous slack. */
struct GcraPolicy {
    uint32_t rate_per_sec = 0;
    uint32_t burst = 0;

    bool enabled() const { return rate_per_sec > 0 && burst > 0; }

    /*** emission interval in ms, rounded up so the effective rate is never
     * more generous than the configured one */
    int64_t interval_ms() const {
        return (1000 + static_cast<int64_t>(rate_per_sec) - 1) / rate_per_sec;
    }

    /*** burst tolerance: (burst-1)*T, so `burst` is the exact instantaneous
     * capacity (burst instantaneous admissions conform, burst+1 rejects) */
    int64_t burst_tolerance_ms() const {
        return (static_cast<int64_t>(burst) - 1) * interval_ms();
    }
};

struct GcraDecision {
    bool allowed = false;
    int64_t retry_after_ms = 0;  // exact; >= 1 whenever rejected
};

/*** One-flow GCRA admission on a caller-owned TAT cell.
 * CAS loop: safe under any interleaving (also fine under an external lock,
 * which is how ShardedIpLimiter uses it — the atomic keeps the free function
 * reusable for a future lock-free shard). */
inline GcraDecision gcra_admit(std::atomic<int64_t>& tat_ms, const GcraPolicy& policy,
                               int64_t now_ms) {
    if (!policy.enabled()) {
        return {true, 0};
    }
    const int64_t interval = policy.interval_ms();
    const int64_t sigma = policy.burst_tolerance_ms();
    int64_t tat = tat_ms.load(std::memory_order_relaxed);
    for (;;) {
        const int64_t effective = std::max(tat, now_ms);
        if (effective - now_ms > sigma) {
            // rejected: next conforming time is TAT - sigma
            return {false, effective - sigma - now_ms};
        }
        if (tat_ms.compare_exchange_weak(tat, effective + interval,
                                         std::memory_order_relaxed)) {
            return {true, 0};
        }
    }
}

/*** Identity of one metering source: address family + normalized prefix.
 * IPv4 keeps its 4 bytes in prefix[0..3]; IPv6 is pre-folded to /64 (8 bytes)
 * by subject.h — a user rotating addresses inside their /64 must not mint
 * fresh identities. Fixed 17 bytes, cheap to hash and compare. */
struct SubjectKey {
    uint8_t family = 4;
    std::array<uint8_t, 8> prefix{};

    bool operator==(const SubjectKey& other) const {
        return family == other.family && prefix == other.prefix;
    }
};

struct SubjectHash {
    /*** FNV-1a over the 17 bytes with a 64-bit finalizer mix: shard index and
     * bucket both derive from this, so the low bits must not cluster. */
    size_t operator()(const SubjectKey& key) const {
        uint64_t h = 1469598103934665603ULL;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&key);
        for (size_t i = 0; i < sizeof(SubjectKey); ++i) {
            h ^= bytes[i];
            h *= 1099511628211ULL;
        }
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }
};

/*** Memory-bounded sharded GCRA store. All admissions for one key serialize
 * on that key's shard mutex; different keys proceed in parallel. `now_ms` is
 * injected by the caller (steady-clock at the wiring layer), which is what
 * makes the sweep/expiry logic exactly unit-testable. */
class ShardedIpLimiter {
  public:
    static constexpr size_t kShards = 16;

    struct Stats {
        size_t tracked = 0;        // live cells across shards
        uint64_t rejects = 0;      // policy rejections (capacity excluded)
        uint64_t evictions = 0;    // idle-sweep + at-capacity reclaims
        uint64_t saturations = 0;  // new sources rejected because at capacity
        uint64_t fail_opens = 0;   // degenerate config or allocation failure
    };

    ShardedIpLimiter(GcraPolicy policy, size_t max_tracked)
        : policy_(policy), max_tracked_(max_tracked) {}

    ShardedIpLimiter(const ShardedIpLimiter&) = delete;
    ShardedIpLimiter& operator=(const ShardedIpLimiter&) = delete;

    /*** Meter one admission for `key`. Never blocks beyond one shard mutex;
     * never allocates after the cell exists. */
    GcraDecision admit(const SubjectKey& key, int64_t now_ms) {
        if (!policy_.enabled()) {
            return {true, 0};
        }
        const size_t shard_idx = SubjectHash{}(key) % kShards;
        Shard& shard = shards_[shard_idx];
        std::lock_guard<std::mutex> lock(shard.mu);
        auto& map = shard.map;
        auto it = map.find(key);
        if (it != map.end()) {
            it->second.last_seen_ms = now_ms;
            const auto decision = gcra_admit(it->second.tat_ms, policy_, now_ms);
            if (!decision.allowed) {
                rejects_.fetch_add(1, std::memory_order_relaxed);
            }
            return decision;
        }
        if (per_shard_budget() == 0) {
            // degenerate configuration: track nothing, fail open
            fail_opens_.fetch_add(1, std::memory_order_relaxed);
            return {true, 0};
        }
        if (map.size() >= per_shard_budget()) {
            sweep_idle_locked(map, now_ms);
        }
        if (map.size() >= per_shard_budget()) {
            // table full of live sources: keep protecting the tracked ones,
            // shed the unknown newcomer (retry: a slot may free up)
            saturations_.fetch_add(1, std::memory_order_relaxed);
            rejects_.fetch_add(1, std::memory_order_relaxed);
            return {false, policy_.interval_ms()};
        }
        try {
            // in-place default construction: Entry is not copyable (atomic)
            auto result = map.try_emplace(key);
            Entry& entry = result.first->second;
            entry.last_seen_ms = now_ms;
            const auto decision = gcra_admit(entry.tat_ms, policy_, now_ms);
            // a fresh cell (tat == 0) always conforms
            if (!decision.allowed) {
                rejects_.fetch_add(1, std::memory_order_relaxed);
            }
            return decision;
        } catch (const std::bad_alloc&) {
            fail_opens_.fetch_add(1, std::memory_order_relaxed);
            return {true, 0};
        }
    }

    Stats stats() const {
        Stats out;
        out.rejects = rejects_.load(std::memory_order_relaxed);
        out.evictions = evictions_.load(std::memory_order_relaxed);
        out.saturations = saturations_.load(std::memory_order_relaxed);
        out.fail_opens = fail_opens_.load(std::memory_order_relaxed);
        for (const auto& shard : shards_) {
            std::lock_guard<std::mutex> lock(shard.mu);
            out.tracked += shard.map.size();
        }
        return out;
    }

  private:
    struct Entry {
        // atomic for symmetry with the free gcra_admit(); all accesses of a
        // cell happen under its shard mutex
        std::atomic<int64_t> tat_ms{0};
        int64_t last_seen_ms = 0;
    };

    struct Shard {
        mutable std::mutex mu;
        std::unordered_map<SubjectKey, Entry, SubjectHash> map;
    };

    size_t per_shard_budget() const {
        if (max_tracked_ == 0) {
            // degenerate configuration: track nothing, fail open (see header)
            return 0;
        }
        return std::max<size_t>(max_tracked_ / kShards, 1);
    }

    /*** Reclaim entries idle beyond 2*(σ+T): the bucket has fully drained and
     * the source stayed silent. Dropping such state only forgives debt the
     * source no longer has, so a below-rate client is never punished by
     * eviction. O(shard) on the insert path only when the shard is at budget. */
    void sweep_idle_locked(std::unordered_map<SubjectKey, Entry, SubjectHash>& map,
                           int64_t now_ms) {
        const int64_t idle = 2 * (policy_.burst_tolerance_ms() + policy_.interval_ms());
        for (auto it = map.begin(); it != map.end();) {
            if (now_ms - it->second.last_seen_ms > idle) {
                it = map.erase(it);
                evictions_.fetch_add(1, std::memory_order_relaxed);
            } else {
                ++it;
            }
        }
    }

    GcraPolicy policy_;
    size_t max_tracked_;
    std::array<Shard, kShards> shards_;
    std::atomic<uint64_t> rejects_{0};
    std::atomic<uint64_t> evictions_{0};
    std::atomic<uint64_t> saturations_{0};
    std::atomic<uint64_t> fail_opens_{0};
};

}  // namespace ratelimit
}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_RATE_LIMIT_GCRA_H
