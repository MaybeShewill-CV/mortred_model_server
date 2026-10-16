/************************************************
 * Round 6 P1-1: RateLimiter abstraction + GCRA authority
 ************************************************/

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "control/rate_limit/rate_limiter.h"
#include "control/rate_limit/headers.h"
#include "control/api_key_manager.h"

using mortred::control::ratelimit::GcraFlowLimiter;
using mortred::control::ApiKey;
using mortred::control::ApiKeyManager;
using mortred::control::AuthReason;

TEST(RateLimitAuthority, gcra_flow_limiter_rejects_over_budget) {
    GcraFlowLimiter lim(2, 2, "api-key-gcra");
    int64_t now = 1'000'000;
    EXPECT_TRUE(lim.admit("k", now).allowed);
    EXPECT_TRUE(lim.admit("k", now).allowed);
    auto d = lim.admit("k", now);
    EXPECT_FALSE(d.allowed);
    EXPECT_GT(d.retry_after_ms, 0);
    EXPECT_STREQ(lim.layer_tag(), "api-key-gcra");
}

TEST(RateLimitAuthority, authority_default_unified) {
    unsetenv("MORTRED_RATE_LIMIT_AUTHORITY");
    EXPECT_TRUE(mortred::control::ratelimit::authority_is_unified());
    setenv("MORTRED_RATE_LIMIT_AUTHORITY", "legacy", 1);
    EXPECT_FALSE(mortred::control::ratelimit::authority_is_unified());
    unsetenv("MORTRED_RATE_LIMIT_AUTHORITY");
}

TEST(RateLimitAuthority, api_key_qps_uses_gcra) {
    ApiKeyManager mgr;
    auto key = std::make_shared<ApiKey>();
    key->name = "t";
    key->scope = "inference";
    key->rate_limit_qps = 1;
    key->enabled = true;
    auto ok = mgr.consume_rate_limit(key);
    EXPECT_EQ(ok.reason, AuthReason::OK);
    auto limited = mgr.consume_rate_limit(key);
    EXPECT_EQ(limited.reason, AuthReason::RATE_LIMITED);
    EXPECT_GT(limited.retry_after_ms, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
