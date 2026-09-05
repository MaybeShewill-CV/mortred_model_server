/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: api_key_manager_unittest.cc
* Date: 26-8-23
************************************************/

// Unit + concurrency tests for the PRODUCTION ApiKeyManager.
//
// P0-2 regression: authenticate() previously returned a raw pointer into the
// key map while reload()/load() cleared and repopulated it - a concurrent
// reload during the caller's scope/name reads was a use-after-free.
// authenticate() now returns shared_ptr<const ApiKey> so the caller owns the
// key for as long as it reads it. The stress test at the bottom drives
// authenticate against a continuous reload loop: with the old signature it
// crashes under ASan and reports under TSAN; with the new one it is clean.
// Registered with the "sanitizer" ctest label (CI TSAN gate).

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
    EXPECT_EQ(mgr.authenticate("Bearer anything"), nullptr);
}

TEST(api_key_manager, authenticates_valid_bearer) {
    const auto path = write_config(key_entry("alpha", "secret-alpha", "inference", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    EXPECT_EQ(mgr.key_count(), 1u);

    const auto key = mgr.authenticate("Bearer secret-alpha");
    ASSERT_NE(key, nullptr);
    EXPECT_EQ(key->name, "alpha");
    EXPECT_EQ(key->scope, "inference");

    EXPECT_EQ(mgr.authenticate("Bearer wrong-secret"), nullptr);
    EXPECT_EQ(mgr.authenticate("Basic dXNlcjpwYXNz"), nullptr);
    EXPECT_EQ(mgr.authenticate(""), nullptr);
}

TEST(api_key_manager, disabled_key_and_scope_semantics) {
    const auto path = write_config(
        key_entry("off", "secret-off", "inference", false, 0) +
        key_entry("admin", "secret-admin", "admin", true, 0) +
        key_entry("su", "secret-su", "all", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    EXPECT_EQ(mgr.key_count(), 3u);

    EXPECT_EQ(mgr.authenticate("Bearer secret-off"), nullptr);

    const auto admin = mgr.authenticate("Bearer secret-admin");
    ASSERT_NE(admin, nullptr);
    EXPECT_TRUE(ApiKeyManager::has_scope(admin, "admin"));
    EXPECT_FALSE(ApiKeyManager::has_scope(admin, "inference"));

    const auto su = mgr.authenticate("Bearer secret-su");
    ASSERT_NE(su, nullptr);
    EXPECT_TRUE(ApiKeyManager::has_scope(su, "inference"));
    EXPECT_TRUE(ApiKeyManager::has_scope(su, "admin"));

    EXPECT_FALSE(ApiKeyManager::has_scope(nullptr, "admin"));
}

TEST(api_key_manager, rate_limit_rejects_within_window) {
    const auto path = write_config(key_entry("limited", "secret-limited", "inference", true, 2));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));

    int rejected = 0;
    for (int i = 0; i < 50; ++i) {
        if (mgr.authenticate("Bearer secret-limited") == nullptr) {
            ++rejected;
        }
    }
    // qps=2: even if the fixed window boundary is crossed mid-loop, at most
    // a handful pass - the overwhelming majority must be rejected
    EXPECT_GE(rejected, 40);
}

TEST(api_key_manager, reload_swaps_active_keys) {
    const auto path = write_config(key_entry("alpha", "secret-alpha", "inference", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    EXPECT_NE(mgr.authenticate("Bearer secret-alpha"), nullptr);

    // rewrite the same file with a different key set, then hot-reload
    {
        std::ofstream out(path);
        out << key_entry("beta", "secret-beta", "inference", true, 0);
    }
    ASSERT_TRUE(mgr.reload());

    EXPECT_EQ(mgr.authenticate("Bearer secret-alpha"), nullptr);  // old key gone
    const auto beta = mgr.authenticate("Bearer secret-beta");
    ASSERT_NE(beta, nullptr);
    EXPECT_EQ(beta->name, "beta");
    EXPECT_EQ(mgr.key_count(), 1u);
}

TEST(api_key_manager, list_keys_reports_counters_without_hash) {
    const auto path = write_config(key_entry("alpha", "secret-alpha", "inference", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));
    ASSERT_NE(mgr.authenticate("Bearer secret-alpha"), nullptr);

    const auto keys = mgr.list_keys();
    ASSERT_EQ(keys.size(), 1u);
    EXPECT_EQ(keys[0].name, "alpha");
    EXPECT_EQ(keys[0].scope, "inference");
    EXPECT_TRUE(keys[0].enabled);
    EXPECT_GE(keys[0].total_requests, 1u);
    EXPECT_EQ(keys[0].total_rejected, 0u);
    // KeyInfo has no hash field by construction (compile-time guarantee)
}

TEST(api_key_manager, concurrent_reload_never_dangles_authenticate) {
    // THE P0-2 regression test: authenticate() readers race a reload() loop
    // that keeps swapping the whole key set. Every successful authenticate
    // result is fully read (the exact fields the gateway touches) while the
    // reloader runs - with the old raw-pointer return this is the UAF window.
    const auto path = write_config(key_entry("stable", "secret-stable", "inference", true, 0));
    ApiKeyManager mgr;
    ASSERT_TRUE(mgr.load(path));

    // variant B additionally carries a bulk of filler keys: each reload then
    // allocates and frees many ApiKey objects, maximizing heap churn around
    // the readers
    std::string variant_b = key_entry("stable", "secret-stable", "inference", true, 0);
    for (int i = 0; i < 64; ++i) {
        variant_b += key_entry("filler" + std::to_string(i),
                               "filler-secret-" + std::to_string(i), "inference", true, 0);
    }
    const std::string variant_a =
        key_entry("stable", "secret-stable", "inference", true, 0);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> auth_ok{0};

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                const auto key = mgr.authenticate("Bearer secret-stable");
                if (key != nullptr) {
                    // the gateway reads exactly these fields after auth
                    EXPECT_EQ(key->name, "stable");
                    EXPECT_EQ(key->scope, "inference");
                    auth_ok.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    std::thread reloader([&]() {
        bool flip = false;
        while (!stop.load(std::memory_order_relaxed)) {
            {
                std::ofstream out(path);
                out << (flip ? variant_a : variant_b);
            }
            // a reload that reads a half-written file simply fails and keeps
            // the previous set - both outcomes are valid hot-reload states
            (void)mgr.reload();
            flip = !flip;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    stop.store(true);
    reloader.join();
    for (auto& th : readers) {
        th.join();
    }

    EXPECT_GT(auth_ok.load(), 0u);
    // final state remains usable regardless of which variant was last loaded
    const auto key = mgr.authenticate("Bearer secret-stable");
    ASSERT_NE(key, nullptr);
    EXPECT_EQ(key->name, "stable");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
