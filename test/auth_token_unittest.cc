/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: auth_token_unittest.cc
* Date: 26-8-22
************************************************/

#include <gtest/gtest.h>

#include <cstdlib>

#include "common/auth_token.h"
#include "common/listen_policy.h"

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
    EXPECT_FALSE(is_bearer_authorized("whatever", ""));  // empty token never authorizes
    EXPECT_TRUE(is_bearer_authorized("Bearer t", "t"));
    EXPECT_FALSE(is_bearer_authorized("Bearer wrong", "t"));
    EXPECT_FALSE(is_bearer_authorized("", "t"));
}

TEST(listen_policy, default_expose_is_loopback_only) {
    ::unsetenv("MORTRED_EXPOSE");
    EXPECT_EQ(jinq::common::mortred_expose_mode(), "loopback");
    EXPECT_TRUE(jinq::common::listen_host_permitted("127.0.0.1"));
    EXPECT_FALSE(jinq::common::listen_host_permitted("0.0.0.0"));
}

TEST(listen_policy, docker_and_unsafe_allow_wildcard) {
    ::setenv("MORTRED_EXPOSE", "docker", 1);
    EXPECT_TRUE(jinq::common::listen_host_permitted("0.0.0.0"));
    ::setenv("MORTRED_EXPOSE", "UNSAFE", 1);
    EXPECT_TRUE(jinq::common::expose_allows_non_loopback());
    ::setenv("MORTRED_EXPOSE", "edge", 1);
    EXPECT_FALSE(jinq::common::listen_host_permitted("0.0.0.0"));
    ::unsetenv("MORTRED_EXPOSE");
}
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
