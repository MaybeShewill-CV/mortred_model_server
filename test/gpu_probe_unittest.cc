#include <memory>
#include <gtest/gtest.h>
#include "control/gpu_probe.h"

using mortred::control::FakeGpuProbe;
using mortred::control::GpuProbeSample;
using mortred::control::ChainGpuProbe;
using mortred::control::NvmlGpuProbe;
using mortred::control::SmiGpuProbe;

TEST(GpuProbe, fake_probe_returns_canned_fail_closed_when_unavailable) {
    GpuProbeSample canned;
    canned.available = false;
    canned.backend = "fake";
    FakeGpuProbe fake(canned);
    auto s = fake.sample(0);
    EXPECT_FALSE(s.available);
    EXPECT_EQ(s.backend, "fake");
}

TEST(GpuProbe, fake_probe_available_sample) {
    GpuProbeSample canned;
    canned.available = true;
    canned.name = "FakeGPU";
    canned.mem_total_mib = 8192;
    canned.mem_free_mib = 4096;
    canned.mem_used_mib = 4096;
    canned.util = 42;
    FakeGpuProbe fake(canned);
    auto s = fake.sample(10);
    EXPECT_TRUE(s.available);
    EXPECT_EQ(s.name, "FakeGPU");
    EXPECT_EQ(s.util, 42);
    EXPECT_EQ(s.backend, "fake");
}

TEST(GpuProbe, chain_prefers_first_available) {
    GpuProbeSample a;
    a.available = false;
    GpuProbeSample b;
    b.available = true;
    b.name = "second";
    b.backend = "fake";
    std::vector<std::unique_ptr<mortred::control::GpuProbe>> probes;
    probes.push_back(std::make_unique<FakeGpuProbe>(a));
    probes.push_back(std::make_unique<FakeGpuProbe>(b));
    ChainGpuProbe chain(std::move(probes));
    auto s = chain.sample(0);
    EXPECT_TRUE(s.available);
    EXPECT_EQ(s.name, "second");
}

TEST(GpuProbe, nvml_probe_honest_backend_tag) {
    // When libnvidia-ml.so.1 is present and a device answers, available=true.
    // When absent or queries fail, available=false. Either way backend stays "nvml".
    NvmlGpuProbe nvml;
    auto s = nvml.sample(0);
    EXPECT_EQ(s.backend, "nvml");
    if (s.available) {
        EXPECT_FALSE(s.name.empty());
        EXPECT_GT(s.mem_total_mib, 0);
    }
}


TEST(GpuProbe, shared_probe_can_be_replaced_for_test) {
    GpuProbeSample canned;
    canned.available = true;
    canned.name = "SharedFake";
    canned.mem_total_mib = 1024;
    canned.mem_free_mib = 512;
    canned.backend = "fake";
    mortred::control::set_shared_gpu_probe_for_test(std::make_unique<FakeGpuProbe>(canned));
    auto s = mortred::control::shared_gpu_probe().sample(0);
    EXPECT_TRUE(s.available);
    EXPECT_EQ(s.name, "SharedFake");
    auto live = mortred::control::live_snapshot_from_probe(s);
    EXPECT_TRUE(live.available);
    EXPECT_EQ(live.total_mib, 1024);
    EXPECT_EQ(live.free_mib, 512);
    // restore default chain for other tests
    mortred::control::set_shared_gpu_probe_for_test(mortred::control::default_gpu_probe());
}

TEST(GpuProbe, nvml_probe_survives_repeated_sample) {
    NvmlGpuProbe nvml;
    auto a = nvml.sample(0);
    auto b = nvml.sample(0);
    EXPECT_EQ(a.backend, "nvml");
    EXPECT_EQ(b.backend, "nvml");
    EXPECT_EQ(a.available, b.available);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
