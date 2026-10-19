/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: occupancy_gate_unittest.cc
* Date: 26-9-10
************************************************/

#include <gtest/gtest.h>

#include <atomic>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <string>
#include <thread>

#include "control/occupancy_gate.h"

using mortred::control::OccupancyCheckInput;
using mortred::control::OccupancySibling;
using mortred::control::is_occupancy_gate_error;
using mortred::control::occupancy_enforcement_disabled;
using mortred::control::occupancy_gate_error;
using mortred::control::occupancy_ready_for_spawn;
using mortred::control::query_nvidia_smi;

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
        // Cases below target the stamp / budget / fingerprint rules, not the
        // telemetry gate itself. Unit tests run without a GPU probe, so they
        // opt in to the explicit unverified-start override; the dedicated
        // telemetry tests clear it to assert the fail-closed default.
        in.allow_unknown_gpu = true;
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

// P1: when the live GPU probe is unavailable the fingerprint and free-memory
// checks cannot run, so a stamped TensorRT id must not start "verified".
TEST_F(OccupancyGateTest, stamped_trt_fails_closed_when_gpu_telemetry_is_unavailable) {
    OccupancyCheckInput in = trt_base();
    in.allow_unknown_gpu = false;  // the default the supervisor uses
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.live.available = false;
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_TRUE(is_occupancy_gate_error(err)) << err;
    EXPECT_NE(err.find("GPU telemetry unavailable"), std::string::npos) << err;
    EXPECT_NE(err.find("MORTRED_ALLOW_UNKNOWN_GPU=1"), std::string::npos) << err;
}

TEST_F(OccupancyGateTest, explicit_override_allows_unverified_start) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.live.available = false;
    in.allow_unknown_gpu = true;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, non_trt_id_is_unaffected_by_missing_telemetry) {
    OccupancyCheckInput in = trt_base();
    in.allow_unknown_gpu = false;
    in.candidate_is_tensorrt = false;
    in.candidate_id = "MOBILENETV2";
    in.live.available = false;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, policy_off_alone_still_enforces_without_allow) {
    // Dual-gate (P0-3): occupancy_policy=off alone must NOT disable enforcement.
    unsetenv("MORTRED_OCCUPANCY_ENFORCE");
    unsetenv("MORTRED_OCCUPANCY_ALLOW_DISABLE");
    OccupancyCheckInput in = trt_base();
    in.occupancy_policy = "off";
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_TRUE(is_occupancy_gate_error(err)) << err;
}

TEST_F(OccupancyGateTest, policy_off_with_allow_disable_skips_gate) {
    setenv("MORTRED_OCCUPANCY_ALLOW_DISABLE", "1", 1);
    OccupancyCheckInput in = trt_base();
    in.occupancy_policy = "off";
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
    unsetenv("MORTRED_OCCUPANCY_ALLOW_DISABLE");
}

TEST_F(OccupancyGateTest, env_zero_alone_does_not_disable_enforcement) {
    // Dual-gate: MORTRED_OCCUPANCY_ENFORCE=0 alone must NOT disable.
    unsetenv("MORTRED_OCCUPANCY_ALLOW_DISABLE");
    ::setenv("MORTRED_OCCUPANCY_ENFORCE", "0", 1);
    EXPECT_FALSE(occupancy_enforcement_disabled("enforce"));
    OccupancyCheckInput in = trt_base();
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_TRUE(is_occupancy_gate_error(err)) << err;
    unsetenv("MORTRED_OCCUPANCY_ENFORCE");
}

TEST_F(OccupancyGateTest, env_zero_with_allow_disable_skips_gate) {
    ::setenv("MORTRED_OCCUPANCY_ENFORCE", "0", 1);
    ::setenv("MORTRED_OCCUPANCY_ALLOW_DISABLE", "1", 1);
    EXPECT_TRUE(occupancy_enforcement_disabled("enforce"));
    OccupancyCheckInput in = trt_base();
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
    unsetenv("MORTRED_OCCUPANCY_ENFORCE");
    unsetenv("MORTRED_OCCUPANCY_ALLOW_DISABLE");
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

TEST_F(OccupancyGateTest, fingerprint_skipped_without_live_gpu_when_override_set) {
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

TEST_F(OccupancyGateTest, declared_measured_mismatch_fails_with_calibrate_hint) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.gpu_mem_at_workers = 1;
    in.has_measured_gpu_mem_mib = true;
    // tol = max(64, 1800/100) = 64; delta 200 > 64
    in.measured_gpu_mem_mib = 2000;
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_TRUE(is_occupancy_gate_error(err)) << err;
    EXPECT_EQ(err.find("occupancy gate:"), 0u) << err;
    EXPECT_NE(err.find("1800"), std::string::npos) << err;
    EXPECT_NE(err.find("2000"), std::string::npos) << err;
    EXPECT_NE(err.find("mortredctl calibrate"), std::string::npos) << err;
    EXPECT_NE(err.find("--write-pack"), std::string::npos) << err;
}

TEST_F(OccupancyGateTest, declared_measured_within_tolerance_passes) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.gpu_mem_at_workers = 1;
    in.has_measured_gpu_mem_mib = true;
    // tol = 64; delta 64 is not > 64
    in.measured_gpu_mem_mib = 1864;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, no_measured_skips_declared_measured_check) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 1800;
    in.gpu_mem_at_workers = 1;
    in.has_measured_gpu_mem_mib = false;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST_F(OccupancyGateTest, query_nvidia_smi_times_out_on_hang) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("mortred_smi_hang_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const fs::path fake = dir / "nvidia-smi";
    {
        std::ofstream out(fake);
        out << "#!/bin/sh\nexec sleep 60\n";
    }
    fs::permissions(fake, fs::perms::owner_all | fs::perms::group_exec | fs::perms::others_exec);
    const char* old_path = std::getenv("PATH");
    const std::string new_path = dir.string() + ":" + (old_path ? old_path : "");
    ASSERT_EQ(::setenv("PATH", new_path.c_str(), 1), 0);

    const auto t0 = std::chrono::steady_clock::now();
    const auto snap = query_nvidia_smi(/*timeout_ms=*/300);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

    if (old_path) {
        ::setenv("PATH", old_path, 1);
    } else {
        ::unsetenv("PATH");
    }
    fs::remove_all(dir);

    EXPECT_FALSE(snap.available);
    EXPECT_GE(ms, 250) << "probe returned without running the PATH script, took " << ms << "ms";
    EXPECT_LT(ms, 2000) << "probe must return within timeout, took " << ms << "ms";
}

TEST(OccupancyEnforcementDisable, soft_off_alone_does_not_disable) {
    unsetenv("MORTRED_OCCUPANCY_ENFORCE");
    unsetenv("MORTRED_OCCUPANCY_ALLOW_DISABLE");
    EXPECT_FALSE(mortred::control::occupancy_enforcement_disabled("off"));
    setenv("MORTRED_OCCUPANCY_ENFORCE", "0", 1);
    EXPECT_FALSE(mortred::control::occupancy_enforcement_disabled("enforce"));
    setenv("MORTRED_OCCUPANCY_ALLOW_DISABLE", "1", 1);
    EXPECT_TRUE(mortred::control::occupancy_enforcement_disabled("off"));
    unsetenv("MORTRED_OCCUPANCY_ENFORCE");
    unsetenv("MORTRED_OCCUPANCY_ALLOW_DISABLE");
}

TEST_F(OccupancyGateTest, query_nvidia_smi_survives_process_wide_reaper) {
    // A supervisor-style reaper (tight waitpid(-1, WNOHANG) loop) used to win
    // the race for the probe child: the probe's own reap hit ECHILD, the
    // occupancy gate silently degraded to the no-nvidia-smi path (fingerprint
    // and free-VRAM checks skipped = fail open), and a defensive kill() could
    // hit the already-reaped, possibly recycled pid. The probe must survive
    // the thief: a fully read (EOF) output still parses.
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("mortred_smi_steal_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const fs::path fake = dir / "nvidia-smi";
    {
        std::ofstream out(fake);
        out << "#!/bin/sh\necho 'Fake Occupancy GPU, 24564, 20000'\n";
    }
    fs::permissions(fake, fs::perms::owner_all | fs::perms::group_exec | fs::perms::others_exec);
    const char* old_path = std::getenv("PATH");
    const std::string new_path = dir.string() + ":" + (old_path ? old_path : "");
    ASSERT_EQ(::setenv("PATH", new_path.c_str(), 1), 0);

    // pure busy-poll: the zombie window between child exit and the probe's
    // own reap is only microseconds wide, so a polite 200us polling loop can
    // miss it entirely (verified: the fix-vs-nofix difference was luck).
    std::atomic<bool> stealing{true};
    std::thread thief([&stealing]() {
        while (stealing.load(std::memory_order_relaxed)) {
            while (::waitpid(-1, nullptr, WNOHANG) > 0) {
            }
        }
    });

    constexpr int k_probes = 20;
    int available_count = 0;
    for (int i = 0; i < k_probes; ++i) {
        const auto snap = query_nvidia_smi(/*timeout_ms=*/3000);
        if (snap.available) {
            ++available_count;
            EXPECT_EQ(snap.name, "Fake Occupancy GPU");
            EXPECT_EQ(snap.total_mib, 24564);
            EXPECT_EQ(snap.free_mib, 20000);
        }
    }
    stealing.store(false, std::memory_order_relaxed);
    thief.join();

    if (old_path) {
        ::setenv("PATH", old_path, 1);
    } else {
        ::unsetenv("PATH");
    }
    fs::remove_all(dir);

    EXPECT_EQ(available_count, k_probes)
        << "every probe must survive the process-wide reaper";
}

TEST_F(OccupancyGateTest, host_ram_budget_rejects_overflow) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 100;
    in.gpu_memory_total_mib = 8192;
    in.has_rss_mib = true;
    in.rss_mib = 900;
    in.host_memory_total_mib = 1024;
    in.ram_reserve_pct = 15;
    OccupancySibling sib;
    sib.id = "OTHER";
    sib.rss_mib = 100;
    in.running_siblings.push_back(sib);
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_NE(err.find("host RAM"), std::string::npos) << err;
}

TEST_F(OccupancyGateTest, host_ram_budget_allows_fit) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 100;
    in.gpu_memory_total_mib = 8192;
    in.has_rss_mib = true;
    in.rss_mib = 200;
    in.host_memory_total_mib = 4096;
    in.ram_reserve_pct = 15;
    std::string err;
    EXPECT_TRUE(occupancy_ready_for_spawn(in, &err)) << err;
}

TEST_F(OccupancyGateTest, live_mem_available_rejects) {
    OccupancyCheckInput in = trt_base();
    in.has_gpu_mem_mib = true;
    in.gpu_mem_mib = 100;
    in.has_rss_mib = true;
    in.rss_mib = 512;
    in.live_mem_available_mib = 128;
    std::string err;
    EXPECT_FALSE(occupancy_ready_for_spawn(in, &err));
    EXPECT_NE(err.find("MemAvailable"), std::string::npos) << err;
}
