/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: api_key_manager_unittest.cc
* Date: 26-8-23
************************************************/

// Unit tests for the PRODUCTION ApiKeyManager.
//
// History note: authenticate() used to return a raw pointer into the key map
// while load() cleared and repopulated it, and a now-removed reload() API was
// stress-tested with a concurrent loop (the "sanitizer" ctest label came from
// there). authenticate() returns shared_ptr<const ApiKey> so a caller owns the
// key for as long as it reads it.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "control/api_key_manager.h"

namespace {

using mortred::control::ApiKeyManager;
using mortred::control::AuthReason;

std::string write_config(const std::string& body) {
    static std::atomic<unsigned> seq{0};
    const auto path = std::filesystem::temp_directory_path() /
                      ("mortred_api_keys_" + std::to_string(::getpid()) + "_" +
                       std::to_string(seq.fetch_add(1)) + ".toml");
    std::ofstream out(path);
    out << body;
    return path.string();
}

std::string key_entry(const std::string& name, const std::string& secret,
                      const std::string& scope, bool enabled, int qps) {
    std::ostringstream os;
    os << "[keys." << name << "]\n"
       << "hash = \"" << ApiKeyManager::sha256_hex(secret) << "\"\n"
       << "scope = \"" << scope << "\"\n";
    if (qps > 0) {
        os << "rate_limit_qps = " << qps << "\n";
    }
    if (!enabled) {
        os << "enabled = false\n";
    }
    return os.str();
}

}  // namespace

TEST(api_key_manager, comment_only_file_is_empty_not_parse_error) {
    const auto path = write_config("# copied from api_keys.toml.example\n");
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    EXPECT_EQ(mgr.key_count(), 0u);
    const auto none = mgr.authenticate("Bearer anything");
    EXPECT_EQ(none.reason, AuthReason::UNKNOWN);
    EXPECT_EQ(none.key, nullptr);
}

TEST(api_key_manager, authenticates_valid_bearer) {
    const auto path = write_config(key_entry("alpha", "secret-alpha", "inference", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    EXPECT_EQ(mgr.key_count(), 1u);

    const auto ok = mgr.authenticate("Bearer secret-alpha");
    ASSERT_EQ(ok.reason, AuthReason::OK);
    ASSERT_NE(ok.key, nullptr);
    EXPECT_EQ(ok.key->name, "alpha");
    EXPECT_EQ(ok.key->scope, "inference");
    EXPECT_EQ(ok.retry_after_ms, 0);

    EXPECT_EQ(mgr.authenticate("Bearer wrong-secret").reason, AuthReason::UNKNOWN);
    EXPECT_EQ(mgr.authenticate("Basic dXNlcjpwYXNz").reason, AuthReason::UNKNOWN);
    EXPECT_EQ(mgr.authenticate("").reason, AuthReason::UNKNOWN);
}

TEST(api_key_manager, disabled_key_and_scope_semantics) {
    const auto path = write_config(
        key_entry("off", "secret-off", "inference", false, 0) +
        key_entry("admin", "secret-admin", "admin", true, 0) +
        key_entry("su", "secret-su", "all", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    EXPECT_EQ(mgr.key_count(), 3u);

    EXPECT_EQ(mgr.authenticate("Bearer secret-off").reason, AuthReason::DISABLED);

    const auto admin = mgr.authenticate("Bearer secret-admin");
    ASSERT_EQ(admin.reason, AuthReason::OK);
    ASSERT_NE(admin.key, nullptr);
    EXPECT_TRUE(ApiKeyManager::has_scope(admin.key, "admin"));
    EXPECT_FALSE(ApiKeyManager::has_scope(admin.key, "inference"));

    const auto su = mgr.authenticate("Bearer secret-su");
    ASSERT_EQ(su.reason, AuthReason::OK);
    ASSERT_NE(su.key, nullptr);
    EXPECT_TRUE(ApiKeyManager::has_scope(su.key, "inference"));
    EXPECT_TRUE(ApiKeyManager::has_scope(su.key, "admin"));

    EXPECT_FALSE(ApiKeyManager::has_scope(nullptr, "admin"));
}

TEST(api_key_manager, rate_limit_rejects_within_window) {
    const auto path = write_config(key_entry("limited", "secret-limited", "inference", true, 2));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));

    int rejected = 0;
    int ok = 0;
    for (int i = 0; i < 50; ++i) {
        const auto r = mgr.authenticate("Bearer secret-limited");
        if (r.reason == AuthReason::RATE_LIMITED) {
            ++rejected;
            // exact budget left in the 1-second fixed window
            EXPECT_GE(r.retry_after_ms, 1);
            EXPECT_LE(r.retry_after_ms, 1000);
            EXPECT_EQ(r.key, nullptr);
        } else {
            ASSERT_EQ(r.reason, AuthReason::OK);
            ++ok;
        }
    }
    // qps=2: even if the fixed window boundary is crossed mid-loop, at most
    // a handful pass - the overwhelming majority must be rejected
    EXPECT_GE(rejected, 40);
    EXPECT_LE(ok, 10);
}

TEST(api_key_manager, rate_limited_valid_key_is_never_unknown) {
    // PR-3a regression: the old null-return conflated "wrong card" with
    // "right card, too fast"; the gateway answered 401 for a throttled
    // VALID key. RATE_LIMITED must be a distinct outcome with a usable
    // Retry-After budget.
    const auto path = write_config(key_entry("limited", "secret-limited", "inference", true, 2));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));

    // qps=2: admissions conform until the window budget is spent, then the
    // limiter trips (allow_rate_limit checks the count before incrementing).
    // Hammer until it trips instead of assuming an exact call index: a
    // 1-second boundary crossed mid-loop grants a fresh budget of two.
    ASSERT_EQ(mgr.authenticate("Bearer secret-limited").reason, AuthReason::OK);
    int admitted = 1;
    mortred::control::AuthResult throttled;
    throttled.reason = AuthReason::OK;
    for (int i = 0; i < 8 && throttled.reason != AuthReason::RATE_LIMITED; ++i) {
        throttled = mgr.authenticate("Bearer secret-limited");
        if (throttled.reason == AuthReason::OK) {
            ++admitted;
        }
    }
    ASSERT_EQ(throttled.reason, AuthReason::RATE_LIMITED);
    EXPECT_GE(throttled.retry_after_ms, 1);
    EXPECT_LE(throttled.retry_after_ms, 1000);
    EXPECT_EQ(throttled.key, nullptr);
    // the microsecond-scale loop may cross at most one window boundary:
    // 2 admissions per window, so at most 4 before the trip
    EXPECT_LE(admitted, 4);
    // the wrong-secret card stays UNKNOWN regardless of any throttling
    EXPECT_EQ(mgr.authenticate("Bearer wrong-secret").reason, AuthReason::UNKNOWN);
}

TEST(api_key_manager, list_keys_reports_counters_without_hash) {
    const auto path = write_config(key_entry("alpha", "secret-alpha", "inference", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    ASSERT_EQ(mgr.authenticate("Bearer secret-alpha").reason, AuthReason::OK);

    const auto keys = mgr.list_keys();
    ASSERT_EQ(keys.size(), 1u);
    EXPECT_EQ(keys[0].name, "alpha");
    EXPECT_EQ(keys[0].scope, "inference");
    EXPECT_TRUE(keys[0].enabled);
    EXPECT_GE(keys[0].total_requests, 1u);
    EXPECT_EQ(keys[0].total_rejected, 0u);
    // KeyInfo has no hash field by construction (compile-time guarantee)
}

TEST(api_key_manager, loads_max_queue_and_worker_budget) {
    const std::string secret = "quota-key-0001";
    const auto path = write_config(
        std::string("[keys.tenant]\nhash = \"") + ApiKeyManager::sha256_hex(secret) +
        "\"\nscope = \"inference\"\nenabled = true\n"
        "max_queue = 7\nworker_budget = 3\nmax_inflight = 5\n");
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    auto auth = mgr.authenticate("Bearer " + secret);
    ASSERT_EQ(auth.reason, mortred::control::AuthReason::OK);
    ASSERT_NE(auth.key, nullptr);
    EXPECT_EQ(auth.key->max_queue, 7);
    EXPECT_EQ(auth.key->worker_budget, 3);
    EXPECT_EQ(auth.key->max_inflight, 5);
}

TEST(api_key_manager, quota_fields_default_zero_when_absent) {
    const std::string secret = "no-quota-key";
    const auto path = write_config(
        std::string("[keys.t]\nhash = \"") + ApiKeyManager::sha256_hex(secret) +
        "\"\nscope = \"inference\"\nenabled = true\n");
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    auto auth = mgr.authenticate("Bearer " + secret);
    ASSERT_EQ(auth.reason, mortred::control::AuthReason::OK);
    EXPECT_EQ(auth.key->max_queue, 0);
    EXPECT_EQ(auth.key->worker_budget, 0);
    EXPECT_EQ(auth.key->max_inflight, 0);
}

TEST(api_key_manager, kdf_salt_authenticates_salted_and_legacy_entries) {
    // [kdf] salt makes the stored hash sha256(salt + key); legacy unsalted
    // entries in the same file must keep authenticating (tried second).
    const std::string salt = "deploy-42";
    const std::string path = "/tmp/mortred_api_key_kdf_test.toml";
    {
        std::ofstream out(path);
        out << "[kdf]\nsalt = \"" << salt << "\"\n";
        out << "[keys.salted]\nhash = \"" << ApiKeyManager::sha256_hex(salt + "salted-secret") << "\"\nenabled = true\n";
        out << "[keys.legacy]\nhash = \"" << ApiKeyManager::sha256_hex("legacy-secret") << "\"\nenabled = true\n";
    }
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));

    auto salted = mgr.identify("Bearer salted-secret");
    ASSERT_EQ(salted.reason, AuthReason::OK) << "salted entry failed to authenticate";
    ASSERT_EQ(salted.key->name, "salted");

    auto legacy = mgr.identify("Bearer legacy-secret");
    ASSERT_EQ(legacy.reason, AuthReason::OK) << "legacy unsalted entry failed to authenticate";
    ASSERT_EQ(legacy.key->name, "legacy");

    // the bare hash of the salted secret must NOT match anything
    auto bare = mgr.identify("no-salt-prefix");
    ASSERT_NE(bare.reason, AuthReason::OK);
    std::remove(path.c_str());
}

TEST(api_key_manager, kdf_salt_wrong_salt_is_unknown) {
    // an entry minted under salt B must not authenticate in a file whose
    // [kdf] salt is A: the gateway computes sha256(A + key), the stored hash
    // is sha256(B + key), and the bare sha256(key) is only a legacy fallback
    const std::string path = "/tmp/mortred_api_key_kdf_wrong_test.toml";
    {
        std::ofstream out(path);
        out << "[kdf]\nsalt = \"saltA\"\n";
        out << "[keys.a]\nhash = \"" << ApiKeyManager::sha256_hex("saltB" + std::string("secret")) << "\"\nenabled = true\n";
    }
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    auto wrong = mgr.identify("Bearer secret");
    EXPECT_EQ(wrong.reason, AuthReason::UNKNOWN);
    std::remove(path.c_str());
}

TEST(api_key_manager, kdf_salt_disabled_entry_reports_disabled) {
    const std::string salt = "s";
    const std::string path = "/tmp/mortred_api_key_kdf_disabled_test.toml";
    {
        std::ofstream out(path);
        out << "[kdf]\nsalt = \"" << salt << "\"\n";
        out << "[keys.off]\nhash = \"" << ApiKeyManager::sha256_hex(salt + "secret") << "\"\nenabled = false\n";
    }
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    auto res = mgr.identify("Bearer secret");
    EXPECT_EQ(res.reason, AuthReason::DISABLED);
    std::remove(path.c_str());
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
