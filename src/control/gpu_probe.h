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
#include <cstring>
#include <dlfcn.h>
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

/*** NVML probe via dlopen("libnvidia-ml.so.1"). No hard link required.
 * Init/device query failure → available=false (honest fail-closed) so
 * ChainGpuProbe can fall back to SmiGpuProbe. */
class NvmlGpuProbe final : public GpuProbe {
  public:
    GpuProbeSample sample(int /*timeout_ms*/) override {
        GpuProbeSample out;
        out.backend = "nvml";
        void* handle = ::dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (handle == nullptr) {
            return out;
        }
        using nvmlReturn_t = int;
        using nvmlDevice_t = void*;
        struct nvmlMemory_t { unsigned long long total, free, used; };
        struct nvmlUtilization_t { unsigned int gpu, memory; };
        constexpr int NVML_SUCCESS = 0;
        constexpr int NVML_TEMPERATURE_GPU = 0;

        auto sym = [&](const char* name) { return ::dlsym(handle, name); };
        auto* nvmlInit = reinterpret_cast<nvmlReturn_t (*)()>(sym("nvmlInit_v2"));
        if (nvmlInit == nullptr) {
            nvmlInit = reinterpret_cast<nvmlReturn_t (*)()>(sym("nvmlInit"));
        }
        auto* nvmlShutdown = reinterpret_cast<nvmlReturn_t (*)()>(sym("nvmlShutdown"));
        auto* nvmlDeviceGetHandleByIndex = reinterpret_cast<nvmlReturn_t (*)(unsigned int, nvmlDevice_t*)>(
            sym("nvmlDeviceGetHandleByIndex_v2"));
        if (nvmlDeviceGetHandleByIndex == nullptr) {
            nvmlDeviceGetHandleByIndex = reinterpret_cast<nvmlReturn_t (*)(unsigned int, nvmlDevice_t*)>(
                sym("nvmlDeviceGetHandleByIndex"));
        }
        auto* nvmlDeviceGetName =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int)>(sym("nvmlDeviceGetName"));
        auto* nvmlDeviceGetMemoryInfo =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*)>(sym("nvmlDeviceGetMemoryInfo"));
        auto* nvmlDeviceGetUtilizationRates =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*)>(
                sym("nvmlDeviceGetUtilizationRates"));
        auto* nvmlDeviceGetTemperature =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, int, unsigned int*)>(
                sym("nvmlDeviceGetTemperature"));
        auto* nvmlDeviceGetPowerUsage =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*)>(sym("nvmlDeviceGetPowerUsage"));

        if (nvmlInit == nullptr || nvmlDeviceGetHandleByIndex == nullptr ||
            nvmlDeviceGetName == nullptr || nvmlDeviceGetMemoryInfo == nullptr) {
            ::dlclose(handle);
            return out;
        }
        if (nvmlInit() != NVML_SUCCESS) {
            ::dlclose(handle);
            return out;
        }
        nvmlDevice_t dev = nullptr;
        if (nvmlDeviceGetHandleByIndex(0, &dev) != NVML_SUCCESS || dev == nullptr) {
            if (nvmlShutdown) nvmlShutdown();
            ::dlclose(handle);
            return out;
        }
        char name[96];
        std::memset(name, 0, sizeof(name));
        if (nvmlDeviceGetName(dev, name, sizeof(name)) != NVML_SUCCESS) {
            if (nvmlShutdown) nvmlShutdown();
            ::dlclose(handle);
            return out;
        }
        nvmlMemory_t mem{};
        if (nvmlDeviceGetMemoryInfo(dev, &mem) != NVML_SUCCESS) {
            if (nvmlShutdown) nvmlShutdown();
            ::dlclose(handle);
            return out;
        }
        out.available = true;
        out.name = name;
        out.mem_total_mib = static_cast<int32_t>(mem.total / (1024ull * 1024ull));
        out.mem_free_mib = static_cast<int32_t>(mem.free / (1024ull * 1024ull));
        out.mem_used_mib = static_cast<int32_t>(mem.used / (1024ull * 1024ull));
        if (nvmlDeviceGetUtilizationRates != nullptr) {
            nvmlUtilization_t util{};
            if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
                out.util = static_cast<int32_t>(util.gpu);
            }
        }
        if (nvmlDeviceGetTemperature != nullptr) {
            unsigned int temp = 0;
            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                out.temp_c = static_cast<int32_t>(temp);
            }
        }
        if (nvmlDeviceGetPowerUsage != nullptr) {
            unsigned int mw = 0;
            if (nvmlDeviceGetPowerUsage(dev, &mw) == NVML_SUCCESS) {
                out.power_w = static_cast<double>(mw) / 1000.0;
            }
        }
        if (nvmlShutdown) nvmlShutdown();
        ::dlclose(handle);
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

inline std::unique_ptr<GpuProbe>& shared_gpu_probe_slot() {
    static std::unique_ptr<GpuProbe> probe = default_gpu_probe();
    return probe;
}

/*** Shared process-wide probe for occupancy + sampler + supervisor. */
inline GpuProbe& shared_gpu_probe() {
    auto& slot = shared_gpu_probe_slot();
    if (!slot) {
        slot = default_gpu_probe();
    }
    return *slot;
}

inline void set_shared_gpu_probe_for_test(std::unique_ptr<GpuProbe> p) {
    shared_gpu_probe_slot() = std::move(p);
}

/*** Map probe sample → occupancy LiveGpuSnapshot (fail-closed if unavailable). */
inline LiveGpuSnapshot live_snapshot_from_probe(const GpuProbeSample& s) {
    LiveGpuSnapshot snap;
    if (!s.available) {
        return snap;
    }
    snap.available = true;
    snap.name = s.name;
    snap.total_mib = s.mem_total_mib;
    snap.free_mib = s.mem_free_mib >= 0
                        ? s.mem_free_mib
                        : (s.mem_total_mib >= 0 && s.mem_used_mib >= 0
                               ? s.mem_total_mib - s.mem_used_mib
                               : -1);
    return snap;
}

inline LiveGpuSnapshot query_live_gpu_via_shared_probe(int timeout_ms = 3000) {
    return live_snapshot_from_probe(shared_gpu_probe().sample(timeout_ms));
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_GPU_PROBE_H
