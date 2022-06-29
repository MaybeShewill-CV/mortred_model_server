/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: md5_unittest.cc
* Date: 22-6-2
************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/md5.h"

using mortred::common::Md5;

TEST(md5_unnittest, to_str) {
    std::string test_str_sampe1 = "test";
    EXPECT_STREQ(Md5(test_str_sampe1).to_str().c_str(), "098f6bcd4621d373cade4e832627b4f6");

    std::string test_str_sampe2 = "12349";
    EXPECT_STREQ(Md5(test_str_sampe2).to_str().c_str(), "55d491cf951b1b920900684d71419282");

    std::string test_str_sampe3 = "abc789";
    EXPECT_STREQ(Md5(test_str_sampe3).to_str().c_str(), "440cbbedf1e789ad49ac0969d2d8069a");

    std::string test_str_sampe4 = "abc-123-;]";
    EXPECT_STREQ(Md5(test_str_sampe4).to_str().c_str(), "3fb506dbfeadfe30fafa4a874b168dbe");
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}