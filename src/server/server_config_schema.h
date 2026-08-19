/************************************************
 * Author: Codex
 * File: server_config_schema.h
 *
 * Declarative schema + validator for [*_SERVER] config sections.
 *
 * Policy:
 * - missing required keys / wrong value types -> error (fail-fast);
 * - unknown keys within edit distance 2 of a known key are treated as typos
 *   -> error with a "did you mean" suggestion;
 * - other unknown scalar keys -> warning only (forward compatibility for
 *   test-only / future keys);
 * - `server_url` is accepted as a deprecated alias of `server_uri`.
 ************************************************/

#ifndef MORTRED_SERVER_CONFIG_SCHEMA_H
#define MORTRED_SERVER_CONFIG_SCHEMA_H

#include <algorithm>
#include <string>
#include <vector>

#include <toml/toml.hpp>

namespace jinq {
namespace server {

namespace detail {

/***
 * edit distance between two strings (small inputs; O(n*m) is fine)
 */
inline size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<size_t> prev(b.size() + 1);
    std::vector<size_t> cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) {
        prev[j] = j;
    }
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            cur[j] = std::min(std::min(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
        }
        prev.swap(cur);
    }
    return prev[b.size()];
}

}  // namespace detail

/***
 * Required keys of a [*_SERVER] section (missing -> fail).
 */
inline const std::vector<std::string>& server_required_string_keys() {
    static const std::vector<std::string> k = {"host", "server_uri"};
    return k;
}

inline const std::vector<std::string>& server_required_int_keys() {
    static const std::vector<std::string> k = {"port", "worker_nums"};
    return k;
}

/***
 * Optional string keys. `server_url` is a deprecated alias of `server_uri`.
 */
inline const std::vector<std::string>& server_optional_string_keys() {
    static const std::vector<std::string> k = {
        "auth_token", "server_exe", "stuck_worker_action", "server_url"};
    return k;
}

/***
 * Optional integer keys.
 */
inline const std::vector<std::string>& server_optional_int_keys() {
    static const std::vector<std::string> k = {
        "max_connections", "peer_resp_timeout", "request_size_limit",
        "compute_threads", "handler_threads", "model_run_timeout",
        "rate_limit_qps", "stuck_worker_threshold_times"};
    return k;
}

inline bool key_in(const std::string& key, const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (k == key) {
            return true;
        }
    }
    return false;
}

inline std::vector<std::string> all_server_keys() {
    std::vector<std::string> all;
    for (const auto& k : server_required_string_keys()) all.push_back(k);
    for (const auto& k : server_required_int_keys()) all.push_back(k);
    for (const auto& k : server_optional_string_keys()) all.push_back(k);
    for (const auto& k : server_optional_int_keys()) all.push_back(k);
    return all;
}

/***
 * Validate a [*_SERVER] section. Returns false and fills err on: missing
 * required key, wrong type, or a probable typo (edit distance <= 2). Unknown
 * keys that are not typos produce warnings only (forward compatibility).
 */
inline bool validate_server_section(const toml::table& section,
                                    std::string* err,
                                    std::vector<std::string>* warnings = nullptr) {
    auto fail = [err](const std::string& message) {
        if (err != nullptr) {
            *err = message;
        }
        return false;
    };

    // missing required keys
    for (const auto& key : server_required_string_keys()) {
        if (!section.contains(key)) {
            return fail("missing required key '" + key + "'");
        }
    }
    for (const auto& key : server_required_int_keys()) {
        if (!section.contains(key)) {
            return fail("missing required key '" + key + "'");
        }
    }

    const std::vector<std::string> known = all_server_keys();

    // known keys: type checks; unknown scalar keys: typo-or-warn
    for (const auto& [key, value] : section) {
        if (key_in(key, server_required_string_keys()) || key_in(key, server_optional_string_keys())) {
            if (!value.is_string()) {
                return fail("key '" + key + "' must be a string");
            }
            if (key == "stuck_worker_action") {
                const std::string action = value.value_or<std::string>("");
                if (action != "log" && action != "exit") {
                    return fail("key 'stuck_worker_action' must be 'log' or 'exit', got '" + action + "'");
                }
            }
            continue;
        }
        if (key_in(key, server_required_int_keys()) || key_in(key, server_optional_int_keys())) {
            if (!value.is_integer()) {
                return fail("key '" + key + "' must be an integer");
            }
            continue;
        }
        // unknown key
        if (value.is_table() || value.is_array()) {
            // structured values are not part of the flat server schema
            if (warnings != nullptr) {
                warnings->push_back("server section key '" + key + "' is a table/array (not part of the server schema)");
            }
            continue;
        }
        std::string best;
        size_t best_distance = 3;
        for (const auto& k : known) {
            const size_t d = detail::edit_distance(key, k);
            if (d < best_distance) {
                best_distance = d;
                best = k;
            }
        }
        if (best_distance <= 2) {
            return fail("unknown key '" + key + "' in server section (did you mean '" + best + "'?)");
        }
        if (warnings != nullptr) {
            warnings->push_back("unknown key '" + key + "' in server section (ignored)");
        }
    }
    return true;
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_CONFIG_SCHEMA_H
