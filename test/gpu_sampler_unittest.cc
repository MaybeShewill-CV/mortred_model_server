// Unit tests for the HUD sampler's nvidia-smi CSV row parser.

#include <string>

#include <gtest/gtest.h>

#include "control/gpu_sampler.h"

namespace {
constexpr const char* kFullRow =
    "NVIDIA GeForce RTX 2070, 37, 2834, 8192, 50, 19.5, 1500, 4001, 40, 12, 8";
}

TEST(gpu_sampler_parse, parses_the_full_eleven_field_row) {
    // name is fields[0] and is skipped; the remaining ten are the metrics
    const auto s = mortred::control::GpuSampler::parse_sample(kFullRow, 0);
    EXPECT_EQ(s.util, 37);
    EXPECT_EQ(s.mem_used_mib, 2834);
    EXPECT_EQ(s.mem_total_mib, 8192);
    EXPECT_EQ(s.temp_c, 50);
    EXPECT_DOUBLE_EQ(s.power_w, 19.5);
    EXPECT_EQ(s.clocks_sm_mhz, 1500);
    EXPECT_EQ(s.clocks_mem_mhz, 4001);
    EXPECT_EQ(s.fan_pct, 40);
    EXPECT_EQ(s.pcie_gen, 12);
    EXPECT_EQ(s.pcie_width, 8);
    EXPECT_GT(s.t_unix_ms, 0);
}

TEST(gpu_sampler_parse, extracts_trimmed_util_column) {
    std::string util;
    mortred::control::GpuSampler::parse_sample(kFullRow, 0, &util);
    EXPECT_EQ(util, "37");
}

TEST(gpu_sampler_parse, na_fields_fall_back_to_unknown) {
    const auto s = mortred::control::GpuSampler::parse_sample(
        "NVIDIA GeForce RTX 2070, 37, 2834, 8192, [N/A], [N/A], 1500, 4001, [N/A], 12, 8", 0);
    EXPECT_EQ(s.temp_c, -1);
    EXPECT_DOUBLE_EQ(s.power_w, -1.0);
    EXPECT_EQ(s.fan_pct, -1);
    EXPECT_EQ(s.util, 37);
}

TEST(gpu_sampler_parse, short_row_keeps_unknown_sentinels) {
    const auto s = mortred::control::GpuSampler::parse_sample("name, 5", 0);
    EXPECT_EQ(s.util, -1);
    EXPECT_EQ(s.mem_total_mib, -1);
    EXPECT_EQ(s.clocks_sm_mhz, -1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
