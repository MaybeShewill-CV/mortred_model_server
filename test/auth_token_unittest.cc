/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: auth_token_unittest.cc
* Date: 26-8-22
************************************************/

#include <gtest/gtest.h>

#include "common/auth_token.h"

using jinq::common::bearer_token_of;
using jinq::common::constant_time_equals;
using jinq::common::is_bearer_authorized;
using jinq::common::is_loopback_host;

TEST(auth_token, loopback_detection) {
    EXPECT_TRUE(is_loopback_host("localhost"));
    EXPECT_TRUE(is_loopback_host("LOCALHOST"));
    EXPECT_TRUE(is_loopback_host("127.0.0.1"));
    EXPECT_TRUE(is_loopback_host("127.0.0.254"));
    EXPECT_TRUE(is_loopback_host("::1"));
    EXPECT_FALSE(is_loopback_host("0.0.0.0"));
    EXPECT_FALSE(is_loopback_host("127.evil"));
    EXPECT_FALSE(is_loopback_host("192.168.1.1"));
}

TEST(auth_token, bearer_extraction) {
    EXPECT_EQ(bearer_token_of("Bearer abc"), "abc");
    EXPECT_EQ(bearer_token_of("bearer abc"), "abc");
    EXPECT_EQ(bearer_token_of("  BEARER   abc  "), "abc");
    EXPECT_EQ(bearer_token_of("Basic abc"), "");
    EXPECT_EQ(bearer_token_of(""), "");
}

TEST(auth_token, constant_time_compare) {
    EXPECT_TRUE(constant_time_equals("secret", "secret"));
    EXPECT_FALSE(constant_time_equals("secret", "secretx"));
    EXPECT_FALSE(constant_time_equals("", "x"));
    EXPECT_TRUE(constant_time_equals("", ""));
}

TEST(auth_token, authorization_entry_point) {
    EXPECT_TRUE(is_bearer_authorized("whatever", ""));  // token disabled
    EXPECT_TRUE(is_bearer_authorized("Bearer t", "t"));
    EXPECT_FALSE(is_bearer_authorized("Bearer wrong", "t"));
    EXPECT_FALSE(is_bearer_authorized("", "t"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
