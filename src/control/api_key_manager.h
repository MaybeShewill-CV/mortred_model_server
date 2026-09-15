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
    int rate_limit_qps = 0;        // 0 = unlimited
    bool enabled = true;
    // runtime state (not persisted). mutable on purpose: authenticate()
    // hands out const keys (shared_ptr<const ApiKey>) that must still count
    // and rate-limit; everything mutable here is internal synchronization
    // state, never identity/config
    mutable std::atomic<uint64_t> total_requests{0};
    mutable std::atomic<uint64_t> total_rejected{0};
    mutable std::mutex rate_mu;
    mutable int64_t rate_window_start = 0;
    mutable int rate_window_count = 0;
};

/***
 * Manages API keys from conf/api_keys.toml. Keys are stored as SHA-256
 * hashes (never plaintext); the config file is hot-reloadable via
 * reload(). Thread-safe.
 *
 * Concurrency contract (P0-2): reload()/load() may replace the whole key
 * set at any time, so authenticate() returns a shared_ptr granting the
 * CALLER ownership for as long as it reads the key (name/scope/counters).
 * Callers must never store the raw pointer beyond the shared_ptr's
 * lifetime - the old raw-pointer return was a use-after-free under
 * concurrent reload.
 */
/*** Why an authenticate() call failed. UNKNOWN and DISABLED fall through to
 * the caller's legacy-token path (401 when nothing else matches);
 * RATE_LIMITED must NOT: a valid key that is merely throttled answers 429 +
 * Retry-After, never "unauthorized". */
enum class AuthReason { OK, UNKNOWN, DISABLED, RATE_LIMITED };

struct AuthResult {
    std::shared_ptr<const ApiKey> key;  // non-null iff reason == OK
    AuthReason reason = AuthReason::UNKNOWN;
    // exact ms until the per-key 1-second fixed window resets (1..1000);
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
     *   [keys.<name>]
     *   hash = "sha256hex..."   # SHA-256 of the API key string
     *   scope = "inference"
     *   rate_limit_qps = 100
     *   enabled = true
     */
    bool load(const std::string& path);

    /*** reload from the same path (hot reload) */
    bool reload();

    /***
     * Authenticate a request: extract Bearer token, hash it, look up,
     * enforce the per-key rate limit.
     * @return reason plus key ownership for OK / the exact Retry-After
     * budget for RATE_LIMITED (see AuthResult). A throttled VALID key is
     * RATE_LIMITED, not a null-key failure — conflating the two makes the
     * gateway answer 401 for "sent too fast", which no client can act on.
     * The returned shared_ptr keeps the key alive across a concurrent
     * reload() that swaps the whole set (P0-2).
     */
    AuthResult authenticate(const std::string& authorization_header);

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

    /*** compute SHA-256 hex of a string (for generating key hashes) */
    static std::string sha256_hex(const std::string& input);

  private:
    mutable std::mutex mu_;
    std::string config_path_;
    std::unordered_map<std::string, std::shared_ptr<ApiKey>> keys_;  // hash -> key

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
        key->enabled = table.count("enabled") != 0
                            ? mini_toml::to_bool(table.at("enabled"), true)
                            : true;
        if (!key->hash.empty()) {
            keys_[key->hash] = key;
        }
    }
    return true;
}

inline bool ApiKeyManager::reload() {
    if (config_path_.empty()) {
        return false;
    }
    return load(config_path_);
}

inline AuthResult ApiKeyManager::authenticate(
    const std::string& authorization_header) {
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
    // alive through the caller's reads even if reload() swaps the set now
    const std::string hash = sha256_hex(token);
    std::shared_ptr<const ApiKey> key;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = keys_.find(hash);
        if (it == keys_.end()) {
            out.reason = AuthReason::UNKNOWN;
            return out;
        }
        if (!it->second->enabled) {
            out.reason = AuthReason::DISABLED;
            return out;
        }
        key = it->second;
    }

    // rate limit (the local shared_ptr keeps the key alive here)
    key->total_requests.fetch_add(1);
    if (!allow_rate_limit(key.get())) {
        key->total_rejected.fetch_add(1);
        out.reason = AuthReason::RATE_LIMITED;
        // exact budget left in this 1-second fixed window: the limiter
        // buckets on floor(steady_ms / 1000)
        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
        int64_t remaining = 1000 - (now_ms % 1000);
        if (remaining < 1) {
            remaining = 1;
        }
        if (remaining > 1000) {
            remaining = 1000;
        }
        out.retry_after_ms = static_cast<int32_t>(remaining);
        return out;
    }
    out.reason = AuthReason::OK;
    out.key = std::move(key);
    return out;
}

inline bool ApiKeyManager::has_scope(const std::shared_ptr<const ApiKey>& key,
                                     const std::string& required) {
    if (key == nullptr) {
        return false;
    }
    return key->scope == "all" || key->scope == required;
}

inline bool ApiKeyManager::allow_rate_limit(const ApiKey* key) {
    if (key->rate_limit_qps <= 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(key->rate_mu);
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    if (key->rate_window_start != now) {
        key->rate_window_start = now;
        key->rate_window_count = 0;
    }
    if (key->rate_window_count >= key->rate_limit_qps) {
        return false;
    }
    ++key->rate_window_count;
    return true;
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
