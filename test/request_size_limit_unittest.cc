/************************************************
 * Author: Codex
 * File: request_size_limit_unittest.cc
 * Date: 2026-08-13
 ************************************************/

#include <cstddef>

#include <gtest/gtest.h>

#include "common/request_size_limit.h"

using jinq::common::k_default_request_size_limit_mb;

TEST(request_size_limit, default_limit_is_64_mb) {
    EXPECT_EQ(k_default_request_size_limit_mb, static_cast<size_t>(64));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
