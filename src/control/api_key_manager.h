/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: api_key_manager.h
* Date: 26-8-23
************************************************/

#ifndef MORTRED_CONTROL_API_KEY_MANAGER_H
#define MORTRED_CONTROL_API_KEY_MANAGER_H

#include <atomic>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "control/mini_toml.h"
#include "control/rate_limit/gcra.h"
#include "control/rate_limit/rate_limiter.h"

#include <openssl/sha.h>

namespace mortred {
namespace control {

/***
 * One API key: SHA-256 hashed at rest, with scope and quota.
 */
struct ApiKey {
    std::string name;              // display name / tenant identifier
    std::string hash;              // SHA-256 hex of the key string
    std::string scope = "inference";  // inference | admin | all
    int rate_limit_qps = 0;        // 0 = unlimited; enforced as GCRA at gateway
    // Per-tenant concurrent in-flight budget (0 = unlimited).
    // Complements rate_limit_qps (GCRA). Callers acquire before work and
    // release in all exit paths; gateway wiring is the admission point.
    int max_inflight = 0;
    // Round 6 P1-2 tenant quota object (enforced once at gateway).
    int max_queue = 0;
    int worker_budget = 0;
    mutable std::atomic<int> inflight{0};
    mutable std::atomic<int> queued{0};
    mutable std::atomic<uint64_t> reject_queue{0};
    mutable std::atomic<uint64_t> reject_inflight{0};
    mutable std::atomic<uint64_t> reject_qps{0};
    bool enabled = true;
    // runtime state (not persisted). mutable on purpose: authenticate()
    // hands out const keys (shared_ptr<const ApiKey>) that must still count
    // and rate-limit; everything mutable here is internal synchronization
    // state, never identity/config
    mutable std::atomic<uint64_t> total_requests{0};
    mutable std::atomic<uint64_t> total_rejected{0};
    // Round 6 P1-1: per-key GCRA TAT (gateway sole public QPS meter).
    mutable std::atomic<int64_t> rate_tat_ms{0};
};

/***
 * Manages API keys from conf/api_keys.toml. Keys are stored as SHA-256
 * hashes (never plaintext); the config loads once at gateway startup
 * (editing the file requires restarting the gateway child). Thread-safe.
 *
 * Concurrency contract: load() replaces the whole key set, and
 * authenticate()/identify() hand out shared_ptr ownership so a caller
 * reading name/scope/counters can never dangle against a future key-set
 * replacement (the old raw-pointer return was a use-after-free class).
 */
/*** Why an authenticate() call failed. UNKNOWN and DISABLED fall through to
 * the caller's legacy-token path (401 when nothing else matches);
 * RATE_LIMITED must NOT: a valid key that is merely throttled answers 429 +
 * Retry-After, never "unauthorized". */
enum class AuthReason { OK, UNKNOWN, DISABLED, RATE_LIMITED };

struct AuthResult {
    std::shared_ptr<const ApiKey> key;  // non-null iff reason == OK
    AuthReason reason = AuthReason::UNKNOWN;
    // exact ms until the next GCRA emission is allowed (at least 1 when limited);
    // 0 unless reason == RATE_LIMITED
    int32_t retry_after_ms = 0;
};

class ApiKeyManager {
  public:
    /***
     * Load keys from a TOML file. Returns false only when the file cannot be
     * parsed. A readable file with zero [keys.*] entries returns true and
     * key_count()==0 (empty is not a parse error).
     * Format:
     *   [kdf]
     *   salt = "..."            # optional per-file salt; when set, every
     *                           # hash = sha256hex(salt + key). Legacy entries
     *                           # hashed without the salt keep working (the
     *                           # unsalted hash is tried second).
     *   [keys.<name>]
     *   hash = "sha256hex..."   # SHA-256 of the API key string (see [kdf])
     *   scope = "inference"
     *   rate_limit_qps = 100   # GCRA at gateway (Round 6)
     *   max_inflight = 8   # optional per-tenant concurrency budget
     *   max_queue = 32     # optional per-tenant queue residency budget
     *   worker_budget = 4  # optional per-tenant hard worker budget
     *   enabled = true
     */
    bool load(const std::string& path);

    /***
     * Authenticate a request: extract Bearer token, hash it, look up,
     * enforce the per-key rate limit.
     * @return reason plus key ownership for OK / the exact Retry-After
     * budget for RATE_LIMITED (see AuthResult). A throttled VALID key is
     * RATE_LIMITED, not a null-key failure — conflating the two makes the
     * gateway answer 401 for "sent too fast", which no client can act on.
     * The returned shared_ptr keeps the key alive across a concurrent
     * reload() that swaps the whole set (P0-2).
     * rate_window_count increments only for scopes key_may_infer accepts
     * (inference, admin, all). Other scopes do not take a slot.
     */
    AuthResult authenticate(const std::string& authorization_header);

    /***
     * Bearer lookup without the per-key window. Gateway process calls this,
     * then accepts scope, then consume_rate_limit.
     */
    AuthResult identify(const std::string& authorization_header);

    /***
     * Take one per-key window slot. Does not increment rate_window_count
     * unless scope is inference, admin, or all. A full window on an
     * accepted scope is RATE_LIMITED with retry_after_ms and a null key.
     */
    AuthResult consume_rate_limit(const std::shared_ptr<const ApiKey>& key);

    /*** P1-4 first slice: per-tenant in-flight budget helpers. */
    static bool try_acquire_inflight(const ApiKey* key);
    static void release_inflight(const ApiKey* key);
    static bool try_acquire_queue(const ApiKey* key);
    static void release_queue(const ApiKey* key);


    /*** check scope (key ownership is held by the shared_ptr) */
    static bool has_scope(const std::shared_ptr<const ApiKey>& key,
                          const std::string& required);

    /*** list all keys (name, scope, enabled - never the hash) */
    struct KeyInfo {
        std::string name;
        std::string scope;
        bool enabled;
        uint64_t total_requests;
        uint64_t total_rejected;
    };
    std::vector<KeyInfo> list_keys() const;

    /*** number of loaded keys */
    size_t key_count() const;

    /*** Entries into consume_rate_limit. Invalid, disabled, and non-infer
     * bearers must leave this unchanged (ST2). */
    uint64_t consume_calls() const {
        return consume_calls_.load(std::memory_order_relaxed);
    }

    /*** compute SHA-256 hex of a string (for generating key hashes) */
    static std::string sha256_hex(const std::string& input);

  private:
    mutable std::mutex mu_;
    std::string config_path_;
    // per-file KDF salt from [kdf]; empty = legacy unsalted-only file
    std::string kdf_salt_;
    std::unordered_map<std::string, std::shared_ptr<ApiKey>> keys_;  // hash -> key
    std::atomic<uint64_t> consume_calls_{0};

    bool allow_rate_limit(const ApiKey* key);
};

inline bool ApiKeyManager::load(const std::string& path) {
    mini_toml::Doc doc;
    if (!mini_toml::load(path, &doc)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    config_path_ = path;
    keys_.clear();
    kdf_salt_.clear();
    if (doc.count("kdf") != 0) {
        const auto& kdf = doc.at("kdf");
        if (kdf.count("salt") != 0) {
            kdf_salt_ = kdf.at("salt");
        }
    }

    for (const auto& [section, table] : doc) {
        // expect [keys.<name>]
        if (section.rfind("keys.", 0) != 0) {
            continue;
        }
        const std::string name = section.substr(5);
        if (name.empty()) {
            continue;
        }
        auto key = std::make_shared<ApiKey>();
        key->name = name;
        key->hash = table.count("hash") != 0 ? table.at("hash") : "";
        key->scope = table.count("scope") != 0 ? table.at("scope") : "inference";
        key->rate_limit_qps = table.count("rate_limit_qps") != 0
                                  ? mini_toml::to_int(table.at("rate_limit_qps"), 0)
                                  : 0;
        key->max_inflight = table.count("max_inflight") != 0
                                ? mini_toml::to_int(table.at("max_inflight"), 0)
                                : 0;
        key->max_queue = table.count("max_queue") != 0
                             ? mini_toml::to_int(table.at("max_queue"), 0)
                             : 0;
        key->worker_budget = table.count("worker_budget") != 0
                                 ? mini_toml::to_int(table.at("worker_budget"), 0)
                                 : 0;
        key->enabled = table.count("enabled") != 0
                            ? mini_toml::to_bool(table.at("enabled"), true)
                            : true;
        if (!key->hash.empty()) {
            keys_[key->hash] = key;
        }
    }
    return true;
}

inline AuthResult ApiKeyManager::identify(const std::string& authorization_header) {
    AuthResult out;
    // extract Bearer token
    const std::string prefix = "bearer ";
    std::string lower;
    std::transform(authorization_header.begin(), authorization_header.end(),
                   std::back_inserter(lower), [](unsigned char c) { return std::tolower(c); });
    if (lower.rfind(prefix, 0) != 0) {
        out.reason = AuthReason::UNKNOWN;
        return out;
    }
    std::string token = authorization_header.substr(prefix.size());
    // trim
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) {
        token.erase(0, 1);
    }
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
        token.pop_back();
    }
    if (token.empty()) {
        out.reason = AuthReason::UNKNOWN;
        return out;
    }

    // hash and look up; the map holds shared_ptr so the returned key stays
    // alive through the caller's reads even if reload() swaps the set now.
    // A file-level [kdf] salt makes a stolen config resistant to offline
    // per-deployment brute force; legacy unsalted entries keep working (the
    // salted hash is tried first, the bare hash second).
    std::shared_ptr<const ApiKey> key;
    {
        std::lock_guard<std::mutex> lock(mu_);
        // returns the entry matching the hash, or null; a matched-but-disabled
        // entry is returned non-null and reported DISABLED after the lock
        const auto find_by_hash = [&](const std::string& hash) {
            const auto it = keys_.find(hash);
            return it == keys_.end() ? std::shared_ptr<const ApiKey>{} : it->second;
        };
        if (!kdf_salt_.empty()) {
            key = find_by_hash(sha256_hex(kdf_salt_ + token));
        }
        if (key == nullptr) {
            key = find_by_hash(sha256_hex(token));
        }
        if (key == nullptr) {
            out.reason = AuthReason::UNKNOWN;
            return out;
        }
    }
    if (!key->enabled) {
        out.reason = AuthReason::DISABLED;
        return out;
    }
    key->total_requests.fetch_add(1);
    out.reason = AuthReason::OK;
    out.key = std::move(key);
    return out;

    key->total_requests.fetch_add(1);
    out.reason = AuthReason::OK;
    out.key = std::move(key);
    return out;
}

inline AuthResult ApiKeyManager::consume_rate_limit(const std::shared_ptr<const ApiKey>& key) {
    consume_calls_.fetch_add(1, std::memory_order_relaxed);
    AuthResult out;
    if (!key) {
        out.reason = AuthReason::UNKNOWN;
        return out;
    }
    if (!allow_rate_limit(key.get())) {
        key->total_rejected.fetch_add(1);
        key->reject_qps.fetch_add(1, std::memory_order_relaxed);
        out.reason = AuthReason::RATE_LIMITED;
        // GCRA retry: TAT - sigma - now (recompute without mutating on reject path
        // already performed by allow_rate_limit). Approximate with one interval.
        const uint32_t rate = static_cast<uint32_t>(key->rate_limit_qps);
        mortred::control::ratelimit::GcraPolicy policy{rate, rate};
        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
        const int64_t tat = key->rate_tat_ms.load(std::memory_order_relaxed);
        const int64_t sigma = policy.burst_tolerance_ms();
        int64_t remaining = tat - sigma - now_ms;
        if (remaining < 1) {
            remaining = policy.interval_ms();
        }
        if (remaining < 1) {
            remaining = 1;
        }
        out.retry_after_ms = static_cast<int32_t>(remaining);
        return out;
    }
    out.reason = AuthReason::OK;
    out.key = key;
    return out;
}

inline AuthResult ApiKeyManager::authenticate(
    const std::string& authorization_header) {
    AuthResult out = identify(authorization_header);
    if (out.reason != AuthReason::OK || !out.key) {
        return out;
    }
    return consume_rate_limit(out.key);
}

inline bool ApiKeyManager::has_scope(const std::shared_ptr<const ApiKey>& key,
                                     const std::string& required) {
    if (key == nullptr) {
        return false;
    }
    return key->scope == "all" || key->scope == required;
}



inline bool ApiKeyManager::try_acquire_inflight(const ApiKey* key) {
    // Round 8 P1-1: track when worker_budget>0 even if max_inflight<=0.
    // Round 9: enforce worker_budget inside this CAS (same loop as max_inflight)
    // so gateway no longer check-then-act on inflight (TOCTOU-safe).
    if (key == nullptr) {
        return true;
    }
    const bool enforce_inflight = key->max_inflight > 0;
    const bool enforce_budget = key->worker_budget > 0;
    if (!enforce_inflight && !enforce_budget) {
        return true;
    }
    int cur = key->inflight.load(std::memory_order_relaxed);
    while (true) {
        if (enforce_inflight && cur >= key->max_inflight) {
            key->reject_inflight.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (enforce_budget && cur >= key->worker_budget) {
            key->reject_inflight.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (key->inflight.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel)) {
            return true;
        }
    }
}

inline void ApiKeyManager::release_inflight(const ApiKey* key) {
    if (key == nullptr) {
        return;
    }
    if (key->max_inflight <= 0 && key->worker_budget <= 0) {
        return;
    }
    key->inflight.fetch_sub(1, std::memory_order_acq_rel);
}

inline bool ApiKeyManager::try_acquire_queue(const ApiKey* key) {
    if (key == nullptr || key->max_queue <= 0) {
        return true;
    }
    int cur = key->queued.load(std::memory_order_relaxed);
    while (true) {
        if (cur >= key->max_queue) {
            key->reject_queue.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (key->queued.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel)) {
            return true;
        }
    }
}

inline void ApiKeyManager::release_queue(const ApiKey* key) {
    if (key == nullptr || key->max_queue <= 0) {
        return;
    }
    key->queued.fetch_sub(1, std::memory_order_acq_rel);
}

inline bool ApiKeyManager::allow_rate_limit(const ApiKey* key) {
    if (key == nullptr || key->rate_limit_qps <= 0) {
        return true;
    }
    // Same scopes key_may_infer accepts. A rejected scope must not take a
    // slot: the gateway answers the existing 401 and leaves the TAT untouched.
    if (key->scope != "all" && key->scope != "inference" && key->scope != "admin") {
        return true;
    }
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    const uint32_t rate = static_cast<uint32_t>(key->rate_limit_qps);
    mortred::control::ratelimit::GcraPolicy policy{rate, rate};
    const auto decision = mortred::control::ratelimit::gcra_admit(key->rate_tat_ms, policy, now_ms);
    return decision.allowed;
}

inline std::vector<ApiKeyManager::KeyInfo> ApiKeyManager::list_keys() const {
    std::vector<KeyInfo> out;
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [hash, key] : keys_) {
        (void)hash;
        KeyInfo info;
        info.name = key->name;
        info.scope = key->scope;
        info.enabled = key->enabled;
        info.total_requests = key->total_requests.load();
        info.total_rejected = key->total_rejected.load();
        out.push_back(std::move(info));
    }
    return out;
}

inline size_t ApiKeyManager::key_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return keys_.size();
}

inline std::string ApiKeyManager::sha256_hex(const std::string& input) {
    // SHA-256 via OpenSSL (already linked via workflow)
    const unsigned char* data = reinterpret_cast<const unsigned char*>(input.data());
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(data, input.size(), digest);
    char hex[SHA256_DIGEST_LENGTH * 2 + 1] = {0};
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", digest[i]);
    }
    return std::string(hex);
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_API_KEY_MANAGER_H
