/************************************************
 * Author: Codex
 * File: time_stamp_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/time_stamp.h"

using jinq::common::Timestamp;

TEST(time_stamp, invalid_and_now) {
    auto invalid = Timestamp::invalid();
    EXPECT_FALSE(invalid.valid());

    auto now = Timestamp::now();
    EXPECT_TRUE(now.valid());
    EXPECT_GT(now, invalid);
    EXPECT_GT(now.micro_sec_since_epoch(), 0u);
}

TEST(time_stamp, epoch_constructor_and_arithmetic) {
    Timestamp t(1000000); // 1 second since epoch
    EXPECT_EQ(t.micro_sec_since_epoch(), 1000000u);

    EXPECT_EQ((t + uint64_t(500000)).micro_sec_since_epoch(), 1500000u);
    EXPECT_EQ((t + 0.5).micro_sec_since_epoch(), 1500000u);
    EXPECT_EQ((t - 0.5).micro_sec_since_epoch(), 500000u);

    Timestamp high(2000000);
    Timestamp low(1000000);
    EXPECT_DOUBLE_EQ(high - low, 1.0);
    EXPECT_LT(low, high);
    EXPECT_GT(high, low);
    EXPECT_LE(low, high);
    EXPECT_GE(high, low);
    EXPECT_NE(high, low);
}

TEST(time_stamp, to_str_and_format) {
    Timestamp t(123456789); // 123 秒 + 456789 微秒
    EXPECT_EQ(t.to_str(), "123.456789");

    // 格式化输出：%Y-%m-%d 长度为 10，且用 '-' 分隔
    auto date = t.to_format_str("%Y-%m-%d");
    EXPECT_EQ(date.size(), 10u);
    EXPECT_EQ(date[4], '-');
    EXPECT_EQ(date[7], '-');

    auto default_str = t.to_format_str();
    EXPECT_FALSE(default_str.empty());
}
