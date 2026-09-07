/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: trust_tokens_unittest.cc
* Date: 26-9-7
************************************************/

#include <gtest/gtest.h>

#include "control/trust_tokens.h"

using mortred::control::is_scrape_token_error;
using mortred::control::kScrapeTokenCollision;
using mortred::control::kScrapeTokenMissing;
using mortred::control::scrape_token_spawn_error;
using mortred::control::scrape_token_usable;

TEST(TrustTokens, missing_metrics_is_not_usable) {
    const auto st = scrape_token_usable("", "infer", "admin");
    EXPECT_FALSE(st.ok);
    EXPECT_STREQ(st.reason, kScrapeTokenMissing);
    EXPECT_TRUE(is_scrape_token_error(scrape_token_spawn_error(st)));
}

TEST(TrustTokens, distinct_tokens_are_usable) {
    const auto st = scrape_token_usable("scrape", "infer", "admin");
    EXPECT_TRUE(st.ok);
    EXPECT_STREQ(st.reason, "");
}

TEST(TrustTokens, collision_with_infer_is_not_usable) {
    const auto st = scrape_token_usable("same", "same", "admin");
    EXPECT_FALSE(st.ok);
    EXPECT_STREQ(st.reason, kScrapeTokenCollision);
}

TEST(TrustTokens, collision_with_admin_is_not_usable) {
    const auto st = scrape_token_usable("same", "infer", "same");
    EXPECT_FALSE(st.ok);
    EXPECT_STREQ(st.reason, kScrapeTokenCollision);
}

TEST(TrustTokens, empty_infer_does_not_collide) {
    const auto st = scrape_token_usable("scrape", "", "admin");
    EXPECT_TRUE(st.ok);
}

TEST(TrustTokens, empty_admin_does_not_collide) {
    const auto st = scrape_token_usable("scrape", "infer", "");
    EXPECT_TRUE(st.ok);
}

TEST(TrustTokens, spawn_error_prefix_is_stable) {
    EXPECT_FALSE(is_scrape_token_error("executable not found: /tmp/x"));
    EXPECT_FALSE(is_scrape_token_error("TensorRT engine missing or empty: x"));
}
