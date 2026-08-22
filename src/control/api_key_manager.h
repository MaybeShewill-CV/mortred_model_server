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
    // runtime counters (not persisted)
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> total_rejected{0};
    // rate limiter state (simplified fixed window)
    std::mutex rate_mu;
    int64_t rate_window_start = 0;
    int rate_window_count = 0;
};

/***
 * Manages API keys from conf/api_keys.toml. Keys are stored as SHA-256
 * hashes (never plaintext); the config file is hot-reloadable via
 * reload(). Thread-safe.
 */
class ApiKeyManager {
  public:
    /***
     * Load keys from a TOML file. Returns false on parse error.
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
     * Authenticate a request: extract Bearer token, hash it, look up.
     * @return nullptr if not found/disabled; the ApiKey if authorized.
     * Also enforces per-key rate limiting.
     */
    const ApiKey* authenticate(const std::string& authorization_header);

    /*** check scope */
    static bool has_scope(const ApiKey* key, const std::string& required);

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

    bool allow_rate_limit(ApiKey* key);
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
    return !keys_.empty();
}

inline bool ApiKeyManager::reload() {
    if (config_path_.empty()) {
        return false;
    }
    return load(config_path_);
}

inline const ApiKey* ApiKeyManager::authenticate(const std::string& authorization_header) {
    // extract Bearer token
    const std::string prefix = "bearer ";
    std::string lower;
    std::transform(authorization_header.begin(), authorization_header.end(),
                   std::back_inserter(lower), [](unsigned char c) { return std::tolower(c); });
    if (lower.rfind(prefix, 0) != 0) {
        return nullptr;
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
        return nullptr;
    }

    // hash and look up
    const std::string hash = sha256_hex(token);
    std::shared_ptr<ApiKey> key;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = keys_.find(hash);
        if (it == keys_.end() || !it->second->enabled) {
            return nullptr;
        }
        key = it->second;
    }

    // rate limit
    key->total_requests.fetch_add(1);
    if (!allow_rate_limit(key.get())) {
        key->total_rejected.fetch_add(1);
        return nullptr;
    }
    return key.get();
}

inline bool ApiKeyManager::has_scope(const ApiKey* key, const std::string& required) {
    if (key == nullptr) {
        return false;
    }
    return key->scope == "all" || key->scope == required;
}

inline bool ApiKeyManager::allow_rate_limit(ApiKey* key) {
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
