/************************************************
 * File: log_buffer_unittest.cc
 * Round 9 P0-4: LogBuffer must not flush-per-append (stall-safe).
 ************************************************/

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "control/log_buffer.h"

using mortred::control::LogBuffer;

TEST(log_buffer, append_many_lines_completes_without_per_line_fsync_stall) {
    const std::string path = "/tmp/mortred_log_buffer_p04_test.log";
    std::remove(path.c_str());
    {
        LogBuffer buf(path, 10 * 1024 * 1024);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i) {
            buf.append("line-" + std::to_string(i));
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        // Local disk: 1000 appends without per-line flush should be well under 1s.
        EXPECT_LT(ms, 1000);
        EXPECT_GE(buf.size(), 1000u);
    }
    std::ifstream in(path);
    ASSERT_TRUE(in.good());
    std::string last;
    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        ++count;
        last = line;
    }
    EXPECT_GE(count, 64);  // at least one batched flush or dtor flush wrote data
    EXPECT_EQ(last, "line-999");
    std::remove(path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
