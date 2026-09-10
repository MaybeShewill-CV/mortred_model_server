/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: occupancy_gate_unittest.cc
* Date: 26-9-10
************************************************/

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "control/occupancy_gate.h"

using mortred::control::OccupancyCheckInput;
using mortred::control::OccupancySibling;
using mortred::control::is_occupancy_gate_error;
using mortred::control::occupancy_enforcement_disabled;
using mortred::control::occupancy_gate_error;
using mortred::control::occupancy_ready_for_spawn;

namespace {

class OccupancyGateTest : public ::testing::Test {
  protected:
    void SetUp() override {
        previous_ = std::getenv("MORTRED_OCCUPANCY_ENFORCE");
        had_previous_ = previous_ != nullptr;
        if (had_previous_) {
            previous_copy_ = previous_;
        }
        ::unsetenv("MORTRED_OCCUPANCY_ENFORCE");
    }
    void TearDown() override {
        if (had_previous_) {
            ::setenv("MORTRED_OCCUPANCY_ENFORCE", previous_copy_.c_str(), 1);
        } else {
            ::unsetenv("MORTRED_OCCUPANCY_ENFORCE");
        }
    }
    OccupancyCheckInput trt_base() const {
        OccupancyCheckInput in;
        in.pack_active = true;
        in.pack_path = "/tmp/machine-pack.toml";
        in.candidate_id = "YOLOV8";
        in.candidate_is_tensorrt = true;
        in.worker_nums = 1;
        return in;
    }
    const char* previous_ = nullptr;
    bool had_previous_ = false;
    std::string previous_copy_;
};

}  // namespace

TEST_F(OccupancyGateTest, pack_inactive_skips_gate) {
    OccupancyCheckInput in = trt_base();
    in.pack_active = false;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, trt_without_stamp_fails_with_calibrate_command) {
    OccupancyCheckInput in = trt_base();
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_TRUE(is_occupancy_gate_error(err)) << err;
    EXPECT_NE(err.find("mortredctl calibrate --pack /tmp/machine-pack.toml --write-pack"),
              std::string::npos)
        << err;
}

TEST_F(OccupancyGateTest, cpu_or_mnn_without_stamp_passes) {
    OccupancyCheckInput in = trt_base();
    in.candidate_is_tensorrt = false;
    in.candidate_id = "MOBILENETV2";
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, policy_off_allows_trt_without_stamp) {
    OccupancyCheckInput in = trt_base();
    in.occupancy_policy = "off";
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, env_zero_disables_enforcement) {
    ::setenv("MORTRED_OCCUPANCY_ENFORCE", "0", 1);
    EXPECT_TRUE(occupancy_enforcement_disabled("enforce"));
    OccupancyCheckInput in = trt_base();
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, stale_worker_nums_fails) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.gpu_mem_at_workers = 1;
    in.worker_nums = 4;
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_NE(err.find("gpu_mem_at_workers=1"), std::string::npos) << err;
    EXPECT_NE(err.find("calibrate --pack"), std::string::npos) << err;
}

TEST_F(OccupancyGateTest, joint_budget_rejects_overflow) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 2000;
    in.gpu_memory_total_mib = 4096;
    in.gpu_reserve_pct = 15;
    OccupancySibling sib;
    sib.id = "OTHER";
    sib.gpu_mem_mib = 2000;
    in.running_siblings.push_back(sib);
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_NE(err.find("exceeds GPU budget"), std::string::npos) << err;
    EXPECT_NE(err.find("calibrate --pack"), std::string::npos) << err;
}

TEST_F(OccupancyGateTest, joint_budget_allows_fit) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.gpu_mem_at_workers = 1;
    in.gpu_memory_total_mib = 10240;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, fingerprint_skipped_without_live_gpu) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.gpu_name = "NVIDIA GeForce RTX 3080";
    in.gpu_memory_total_mib = 10240;
    in.live.available = false;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, fingerprint_mismatch_fails) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.gpu_name = "NVIDIA GeForce RTX 3080";
    in.live.available = true;
    in.live.name = "NVIDIA GeForce RTX 4090";
    in.live.total_mib = 24564;
    in.live.free_mib = 20000;
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_NE(err.find("fingerprint mismatch"), std::string::npos) << err;
}

TEST_F(OccupancyGateTest, free_memory_shortfall_retries_start_not_calibrate) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 4000;
    in.live.available = true;
    in.live.name = "GPU";
    in.live.total_mib = 8192;
    in.live.free_mib = 512;
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_TRUE(is_occupancy_gate_error(err)) << err;
    EXPECT_NE(err.find("retry start"), std::string::npos) << err;
    EXPECT_EQ(err.find("calibrate --pack"), std::string::npos) << err;
}

TEST_F(OccupancyGateTest, error_prefix) {
    EXPECT_TRUE(is_occupancy_gate_error(occupancy_gate_error("x", "/p.toml")));
    EXPECT_FALSE(is_occupancy_gate_error("fork() failed"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
