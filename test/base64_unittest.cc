/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base64_unittest.cc
* Date: 22-6-2
************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/base64.h"

using ly::common::Base64;

TEST(base64_unnittest, encode) {
    EXPECT_EQ(Base64::base64_encode("abc.txt").size(), 12);

    EXPECT_STREQ(Base64::base64_encode("").c_str(), "");
    EXPECT_STREQ(Base64::base64_encode("f").c_str(), "Zg==");
    EXPECT_STREQ(Base64::base64_encode("fo").c_str(), "Zm8=");
    EXPECT_STREQ(Base64::base64_encode("foo").c_str(), "Zm9v");
    EXPECT_STREQ(Base64::base64_encode("foob").c_str(), "Zm9vYg==");
    EXPECT_STREQ(Base64::base64_encode("fooba").c_str(), "Zm9vYmE=");
    EXPECT_STREQ(Base64::base64_encode("foobar").c_str(), "Zm9vYmFy");
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}