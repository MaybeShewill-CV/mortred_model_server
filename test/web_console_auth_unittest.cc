/************************************************
 * Author: Codex
 * File: web_console_auth_unittest.cc
 * Date: 2026-08-13
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/auth_token.h"

using jinq::common::bearer_token_of;
using jinq::common::constant_time_equals;
using jinq::common::is_bearer_authorized;
using jinq::common::is_loopback_host;

TEST(web_console_auth, loopback_hosts_are_recognized) {
    EXPECT_TRUE(is_loopback_host("127.0.0.1"));
    EXPECT_TRUE(is_loopback_host("127.8.8.8"));
    EXPECT_TRUE(is_loopback_host("localhost"));
    EXPECT_TRUE(is_loopback_host("::1"));
}

TEST(web_console_auth, non_loopback_hosts_are_rejected) {
    EXPECT_FALSE(is_loopback_host("0.0.0.0"));
    EXPECT_FALSE(is_loopback_host("192.168.1.10"));
    EXPECT_FALSE(is_loopback_host(""));
}

TEST(web_console_auth, bearer_token_is_extracted) {
    EXPECT_EQ(bearer_token_of("Bearer secret-token"), "secret-token");
    EXPECT_EQ(bearer_token_of("bearer lower-case-scheme"), "lower-case-scheme");
    EXPECT_EQ(bearer_token_of("Bearer  spaced-token  "), "spaced-token");
}

TEST(web_console_auth, non_bearer_or_malformed_header_yields_empty_token) {
    EXPECT_EQ(bearer_token_of("Basic dXNlcjpwYXNz"), "");
    EXPECT_EQ(bearer_token_of(""), "");
    EXPECT_EQ(bearer_token_of("Bearer"), "");
    EXPECT_EQ(bearer_token_of("Token abc"), "");
}

TEST(web_console_auth, constant_time_equals_compares_exactly) {
    EXPECT_TRUE(constant_time_equals("abc", "abc"));
    EXPECT_FALSE(constant_time_equals("abc", "abd"));
    EXPECT_FALSE(constant_time_equals("abc", "abcd"));
    EXPECT_FALSE(constant_time_equals("", "a"));
    EXPECT_TRUE(constant_time_equals("", ""));
}

TEST(web_console_auth, authorized_requires_matching_token) {
    EXPECT_TRUE(is_bearer_authorized("Bearer s3cret", "s3cret"));
    EXPECT_FALSE(is_bearer_authorized("Bearer wrong", "s3cret"));
    EXPECT_FALSE(is_bearer_authorized("", "s3cret"));
    EXPECT_FALSE(is_bearer_authorized("Basic abc", "s3cret"));
}

TEST(web_console_auth, empty_configured_token_allows_everything) {
    EXPECT_TRUE(is_bearer_authorized("", ""));
    EXPECT_TRUE(is_bearer_authorized("Bearer whatever", ""));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
