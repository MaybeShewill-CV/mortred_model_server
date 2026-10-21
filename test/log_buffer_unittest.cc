/************************************************
 * File: log_buffer_unittest.cc
 * Round 9 P0-4: LogBuffer must not flush-per-append (stall-safe).
 ************************************************/

#include <chrono>
#include <cstdint>
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


TEST(log_buffer, rollover_archives_ended_run_and_starts_fresh) {
    const std::string path = "/tmp/mortred_log_buffer_rollover_test.log";
    std::remove(path.c_str());
    const std::string archive = path + ".run1000";
    std::remove(archive.c_str());
    {
        LogBuffer buf(path, 10 * 1024 * 1024);
        buf.append("crash-evidence-line");
        buf.append("second-line");
        buf.rollover(1000);
        // ring is fresh: the ended run's lines are gone from memory
        EXPECT_EQ(buf.size(), 0u);
        EXPECT_TRUE(buf.slice(0, 10).empty());
        buf.append("new-run-line");
        EXPECT_EQ(buf.size(), 1u);
    }
    // the ended run's content survives in the archive
    std::ifstream in(archive);
    ASSERT_TRUE(in.good()) << "archive file missing";
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("crash-evidence-line"), std::string::npos);
    EXPECT_NE(content.find("second-line"), std::string::npos);
    // the live path holds only the new run
    std::ifstream live(path);
    ASSERT_TRUE(live.good());
    std::string live_content((std::istreambuf_iterator<char>(live)),
                             std::istreambuf_iterator<char>());
    EXPECT_EQ(live_content.find("crash-evidence-line"), std::string::npos);
    EXPECT_NE(live_content.find("new-run-line"), std::string::npos);
    std::remove(archive.c_str());
    std::remove(path.c_str());
}

TEST(log_buffer, rollover_keeps_only_the_newest_k_archives) {
    const std::string path = "/tmp/mortred_log_buffer_sweep_test.log";
    for (int i = 0; i < 6; ++i) {
        std::remove((path + ".run" + std::to_string(100 + i)).c_str());
    }
    std::remove(path.c_str());
    LogBuffer buf(path, 10 * 1024 * 1024);
    for (int i = 0; i < 6; ++i) {
        buf.append("run-" + std::to_string(i));
        buf.rollover(static_cast<uint64_t>(100 + i));
    }
    int archives = 0;
    bool oldest_gone = true;
    int kept_newest = 0;
    for (int i = 0; i < 6; ++i) {
        std::ifstream in(path + ".run" + std::to_string(100 + i));
        if (in.good()) {
            ++archives;
            if (i == 0) {
                oldest_gone = false;  // stamp 100 is the oldest and must be swept
            }
            if (i == 5) {
                kept_newest = 1;
            }
        }
    }
    EXPECT_EQ(archives, static_cast<int>(LogBuffer::kArchiveKeep));
    EXPECT_TRUE(oldest_gone);
    EXPECT_EQ(kept_newest, 1);
    for (int i = 0; i < 6; ++i) {
        std::remove((path + ".run" + std::to_string(100 + i)).c_str());
    }
    std::remove(path.c_str());
}

TEST(log_buffer, rollover_never_touches_size_rotation_suffix) {
    const std::string path = "/tmp/mortred_log_buffer_rot1_test.log";
    const std::string size_rotated = path + ".1";
    std::remove(path.c_str());
    std::remove(size_rotated.c_str());
    {
        std::ofstream old1(size_rotated);
        old1 << "size-rotated-evidence";
    }
    LogBuffer buf(path, 10 * 1024 * 1024);
    buf.append("current-run");
    buf.rollover(5000);
    std::ifstream in(size_rotated);
    ASSERT_TRUE(in.good());
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(content, "size-rotated-evidence");
    std::remove(size_rotated.c_str());
    std::remove((path + ".run5000").c_str());
    std::remove(path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
