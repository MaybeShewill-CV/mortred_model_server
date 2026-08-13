/************************************************
 * Author: Codex
 * File: worker_nums_unittest.cc
 * Date: 2026-08-13
 ************************************************/

#include <gtest/gtest.h>

#include "server/base_server_impl.h"

using jinq::server::parse_worker_nums;

TEST(worker_nums, missing_key_is_rejected) {
    toml::table config;
    EXPECT_EQ(parse_worker_nums(config), -1);
}

TEST(worker_nums, zero_and_negative_are_rejected) {
    toml::table zero_cfg = std::move(toml::parse("worker_nums = 0")).table();
    EXPECT_EQ(parse_worker_nums(zero_cfg), -1);

    toml::table neg_cfg = std::move(toml::parse("worker_nums = -3")).table();
    EXPECT_EQ(parse_worker_nums(neg_cfg), -1);
}

TEST(worker_nums, positive_value_is_kept) {
    toml::table config = std::move(toml::parse("worker_nums = 2")).table();
    EXPECT_EQ(parse_worker_nums(config), 2);
    toml::table config_one = std::move(toml::parse("worker_nums = 1")).table();
    EXPECT_EQ(parse_worker_nums(config_one), 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
