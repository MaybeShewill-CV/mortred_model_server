/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: listen_port_unittest.cc
* Date: 26-10-25
************************************************/

// The listen port is read from the model section or MORTRED_LISTEN_PORT and then
// narrowed to unsigned short; these boundaries are what keeps 0 / 70000 from
// silently becoming a different port.

#include <gtest/gtest.h>

#include "apps/common/listen_port.h"

using mortred::apps::valid_listen_port;

TEST(listen_port, accepts_the_bindable_range) {
    EXPECT_TRUE(valid_listen_port(1));
    EXPECT_TRUE(valid_listen_port(8080));
    EXPECT_TRUE(valid_listen_port(65535));
}

TEST(listen_port, rejects_zero_negative_and_overflow_values) {
    EXPECT_FALSE(valid_listen_port(0));
    EXPECT_FALSE(valid_listen_port(-1));
    EXPECT_FALSE(valid_listen_port(65536));
    EXPECT_FALSE(valid_listen_port(70000));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
