/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: backpressure_unittest.cc
* Date: 26-8-22
************************************************/

#include <gtest/gtest.h>

#include "server/backpressure.h"

using jinq::server::compute_retry_after_seconds;

TEST(backpressure, degenerate_inputs_fall_back_to_one_second) {
    EXPECT_EQ(compute_retry_after_seconds(0, 1000, 4), 1);
    EXPECT_EQ(compute_retry_after_seconds(10, 0, 4), 1);
    EXPECT_EQ(compute_retry_after_seconds(10, 1000, 0), 1);
}

TEST(backpressure, estimates_drain_time_from_ewma) {
    // 10 jobs x 1000ms / 4 workers = 2500ms -> 3s (round up)
    EXPECT_EQ(compute_retry_after_seconds(10, 1000, 4), 3);
    // 100 jobs x 500ms / 2 workers = 25000ms -> 25s
    EXPECT_EQ(compute_retry_after_seconds(100, 500, 2), 25);
}

TEST(backpressure, clamps_to_sixty_seconds) {
    // 100000 x 1000ms / 1 worker would be ~100s
    EXPECT_EQ(compute_retry_after_seconds(100000, 1000, 1), 60);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
