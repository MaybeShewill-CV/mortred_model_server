/************************************************
 * Author: Codex
 * File: rate_limiter_unittest.cc
 * Date: 2026-08-13
 ************************************************/

#include <chrono>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "server/rate_limiter.h"

using jinq::server::FixedWindowRateLimiter;

TEST(rate_limiter, disabled_when_qps_non_positive) {
    FixedWindowRateLimiter limiter(0);
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(limiter.allow("any"));
    }
    limiter.set_max_qps(-1);
    EXPECT_TRUE(limiter.allow("any"));
}

TEST(rate_limiter, allows_exactly_qps_in_window) {
    FixedWindowRateLimiter limiter(3, 200);
    EXPECT_TRUE(limiter.allow("a"));
    EXPECT_TRUE(limiter.allow("a"));
    EXPECT_TRUE(limiter.allow("a"));
    EXPECT_FALSE(limiter.allow("a"));
}

TEST(rate_limiter, different_keys_have_independent_budgets) {
    FixedWindowRateLimiter limiter(1, 200);
    EXPECT_TRUE(limiter.allow("a"));
    EXPECT_FALSE(limiter.allow("a"));
    EXPECT_TRUE(limiter.allow("b"));
}

TEST(rate_limiter, window_resets_after_elapse) {
    FixedWindowRateLimiter limiter(1, 200);
    EXPECT_TRUE(limiter.allow("a"));
    EXPECT_FALSE(limiter.allow("a"));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    EXPECT_TRUE(limiter.allow("a"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
