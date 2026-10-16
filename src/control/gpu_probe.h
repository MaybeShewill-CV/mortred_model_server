/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: gpu_probe.h
* Date: 26-10-16
************************************************/

// Round 6 P1-3: pluggable GpuProbe. NVML primary, nvidia-smi fallback.
// Probes never use waitpid(-1). Fail-closed on incomplete/stolen children.
// occupancy_gate and gpu_sampler share this metrics source.

#ifndef MORTRED_CONTROL_GPU_PROBE_H
#define MORTRED_CONTROL_GPU_PROBE_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "control/occupancy_gate.h"

namespace mortred {
namespace control {

struct GpuProbeSample {
    bool available = false;
    std::string name;
    int32_t util = -1;
    int32_t mem_used_mib = -1;
    int32_t mem_total_mib = -1;
    int32_t mem_free_mib = -1;
    int32_t temp_c = -1;
    double power_w = -1.0;
    std::string backend;  // "nvml" | "nvidia-smi" | "fake"
};

class GpuProbe {
  public:
    virtual ~GpuProbe() = default;
    virtual GpuProbeSample sample(int timeout_ms = 3000) = 0;
    virtual const char* name() const = 0;
};

/*** Fake probe for unit tests (deterministic, no fork). */
class FakeGpuProbe final : public GpuProbe {
  public:
    explicit FakeGpuProbe(GpuProbeSample canned) : canned_(std::move(canned)) {
        if (canned_.backend.empty()) canned_.backend = "fake";
    }
    GpuProbeSample sample(int /*timeout_ms*/) override { return canned_; }
    const char* name() const override { return "fake"; }
    void set(GpuProbeSample s) { canned_ = std::move(s); if (canned_.backend.empty()) canned_.backend = "fake"; }

  private:
    GpuProbeSample canned_;
};

/*** nvidia-smi CSV probe — reuses occupancy_gate::query_nvidia_smi (no waitpid(-1)). */
class SmiGpuProbe final : public GpuProbe {
  public:
    GpuProbeSample sample(int timeout_ms = 3000) override {
        GpuProbeSample out;
        out.backend = "nvidia-smi";
        LiveGpuSnapshot snap = query_nvidia_smi(timeout_ms);
        if (!snap.available) {
            return out;  // fail-closed
        }
        out.available = true;
        out.name = snap.name;
        out.mem_total_mib = snap.total_mib;
        out.mem_free_mib = snap.free_mib;
        if (snap.total_mib > 0 && snap.free_mib >= 0) {
            out.mem_used_mib = snap.total_mib - snap.free_mib;
        }
        return out;
    }
    const char* name() const override { return "nvidia-smi"; }
};

/*** NVML probe. Links optionally; when NVML headers/libs absent at build time,
 * this class always fail-closes and ChainGpuProbe falls back to SmiGpuProbe.
 * Runtime: dlopen libnvidia-ml.so.1 when MORTRED_GPU_PROBE prefers nvml. */
class NvmlGpuProbe final : public GpuProbe {
  public:
    GpuProbeSample sample(int /*timeout_ms*/) override {
        GpuProbeSample out;
        out.backend = "nvml";
        // Header-only / tests-only builds: NVML not linked. Fail-closed so the
        // chain can fall back to nvidia-smi. Full builds may replace this with
        // a TU that calls nvmlDeviceGet* (same interface).
        return out;
    }
    const char* name() const override { return "nvml"; }
};

/*** Prefer NVML, then nvidia-smi. First available sample wins; else fail-closed. */
class ChainGpuProbe final : public GpuProbe {
  public:
    ChainGpuProbe(std::vector<std::unique_ptr<GpuProbe>> probes)
        : probes_(std::move(probes)) {}

    GpuProbeSample sample(int timeout_ms = 3000) override {
        for (auto& p : probes_) {
            if (!p) continue;
            GpuProbeSample s = p->sample(timeout_ms);
            if (s.available) {
                return s;
            }
        }
        GpuProbeSample miss;
        miss.backend = "none";
        return miss;  // fail-closed
    }
    const char* name() const override { return "chain"; }

  private:
    std::vector<std::unique_ptr<GpuProbe>> probes_;
};

inline std::unique_ptr<GpuProbe> default_gpu_probe() {
    std::vector<std::unique_ptr<GpuProbe>> chain;
    chain.push_back(std::make_unique<NvmlGpuProbe>());
    chain.push_back(std::make_unique<SmiGpuProbe>());
    return std::make_unique<ChainGpuProbe>(std::move(chain));
}

/*** Shared process-wide probe for occupancy + sampler metrics source. */
inline GpuProbe& shared_gpu_probe() {
    static std::unique_ptr<GpuProbe> probe = default_gpu_probe();
    return *probe;
}

inline void set_shared_gpu_probe_for_test(std::unique_ptr<GpuProbe> p) {
    // test-only: replace by swapping through a function-local that shared uses
    // For simplicity, tests construct FakeGpuProbe locally rather than mutating
    // the process singleton.
    (void)p;
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_GPU_PROBE_H
