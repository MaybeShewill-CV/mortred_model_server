/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder.cpp
 * Date: 26-9-20
 ************************************************/

#include "models/backend/gpu_jpeg_decoder.h"

#include <chrono>
#include <cstring>
#include <mutex>
#include <vector>

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <opencv2/imgcodecs.hpp>

#include "glog/logging.h"

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {
namespace {

struct NvJpegPlan {
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t state = nullptr;
    cudaStream_t stream = nullptr;
    unsigned char* device_buffer = nullptr;
    size_t device_buffer_bytes = 0;
    unsigned char* pinned_buffer = nullptr;
    size_t pinned_buffer_bytes = 0;
    bool ok = false;
};

bool build_plan(nvjpegBackend_t backend, NvJpegPlan* plan) {
    if (nvjpegCreateEx(backend, nullptr, nullptr, 0, &plan->handle) != NVJPEG_STATUS_SUCCESS ||
        nvjpegJpegStateCreate(plan->handle, &plan->state) != NVJPEG_STATUS_SUCCESS ||
        cudaStreamCreate(&plan->stream) != cudaSuccess) {
        return false;
    }
    plan->ok = true;
    return true;
}

void release_plan(NvJpegPlan* plan) {
    if (plan->state != nullptr) {
        nvjpegJpegStateDestroy(plan->state);
        plan->state = nullptr;
    }
    if (plan->handle != nullptr) {
        nvjpegDestroy(plan->handle);
        plan->handle = nullptr;
    }
    if (plan->stream != nullptr) {
        cudaStreamDestroy(plan->stream);
        plan->stream = nullptr;
    }
    if (plan->device_buffer != nullptr) {
        cudaFree(plan->device_buffer);
        plan->device_buffer = nullptr;
        plan->device_buffer_bytes = 0;
    }
    if (plan->pinned_buffer != nullptr) {
        cudaFreeHost(plan->pinned_buffer);
        plan->pinned_buffer = nullptr;
        plan->pinned_buffer_bytes = 0;
    }
    plan->ok = false;
}

/*** one decode through the given plan; buffer handling mirrors decode() */
bool run_decode(NvJpegPlan* plan, const unsigned char* data, size_t size, int out_w, int out_h) {
    const size_t out_bytes = static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3;
    if (plan->device_buffer_bytes < out_bytes) {
        if (plan->device_buffer != nullptr) {
            cudaFree(plan->device_buffer);
            plan->device_buffer = nullptr;
            plan->device_buffer_bytes = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&plan->device_buffer), out_bytes) != cudaSuccess) {
            return false;
        }
        plan->device_buffer_bytes = out_bytes;
    }
    nvjpegImage_t image{};
    image.channel[0] = plan->device_buffer;
    image.pitch[0] = static_cast<size_t>(out_w) * 3;
    if (nvjpegDecode(plan->handle, plan->state, data, size, NVJPEG_OUTPUT_BGRI, &image, plan->stream) !=
        NVJPEG_STATUS_SUCCESS) {
        return false;
    }
    return cudaStreamSynchronize(plan->stream) == cudaSuccess;
}

/*** representative ~1MP test frame for the capability probe and perf race:
 * gradient + photo-level noise so the compressed size matches real camera
 * output (~0.5 byte/pixel) - a smooth synthetic gradient compresses to a
 * tiny bitstream and makes the GPU path look artificially fast */
std::vector<unsigned char> make_test_jpeg() {
    cv::Mat patch(1024, 1024, CV_8UC3);
    cv::RNG rng(20260920);
    for (int y = 0; y < patch.rows; ++y) {
        auto* row = patch.ptr<unsigned char>(y);
        for (int x = 0; x < patch.cols; ++x) {
            row[x * 3 + 0] = static_cast<unsigned char>((x * 255) / patch.cols);
            row[x * 3 + 1] = static_cast<unsigned char>((y * 255) / patch.rows);
            row[x * 3 + 2] = static_cast<unsigned char>(((x + y) * 127) / patch.cols);
        }
    }
    cv::Mat noise(patch.size(), CV_8UC3);
    rng.fill(noise, cv::RNG::UNIFORM, cv::Scalar(-28, -28, -28), cv::Scalar(28, 28, 28));
    cv::add(patch, noise, patch, cv::noArray(), CV_8UC3);
    std::vector<unsigned char> jpeg;
    const std::vector<int> encode_params{cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", patch, jpeg, encode_params);
    return jpeg;
}

struct ProbeResult {
    bool capable = false;
    bool race_won = false;
    nvjpegBackend_t backend = NVJPEG_BACKEND_GPU_HYBRID;
};

const ProbeResult& probe_once() {
    static const ProbeResult result = [] {
        ProbeResult out;
        const std::vector<unsigned char> jpeg = make_test_jpeg();
        if (jpeg.empty()) {
            return out;
        }
        NvJpegPlan probe_plan;
        bool have_backend = false;
        for (const nvjpegBackend_t candidate : {NVJPEG_BACKEND_HARDWARE, NVJPEG_BACKEND_GPU_HYBRID}) {
            if (!build_plan(candidate, &probe_plan)) {
                release_plan(&probe_plan);
                continue;
            }
            if (run_decode(&probe_plan, jpeg.data(), jpeg.size(), 1024, 1024)) {
                out.capable = true;
                out.backend = candidate;
                have_backend = true;
                break;
            }
            release_plan(&probe_plan);
        }
        if (!have_backend) {
            return out;
        }
        // perf race, production-shaped: the GPU path decodes full frames, so
        // the CPU opponent must use the DCT-reduced decode the production
        // selector would pick for an over-sized image (not the full-decode
        // strawman). GPU must still win by >=20% to take "auto" - consumer
        // cards without the dedicated JPEG engine usually lose here.
        const int rounds = 3;
        double gpu_best = 1e9;
        for (int i = 0; i < rounds; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!run_decode(&probe_plan, jpeg.data(), jpeg.size(), 1024, 1024)) {
                release_plan(&probe_plan);
                return out;
            }
            const auto t1 = std::chrono::steady_clock::now();
            gpu_best = std::min(gpu_best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        release_plan(&probe_plan);
        double cpu_best = 1e9;
        for (int i = 0; i < rounds; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            const cv::Mat decoded = cv::imdecode(jpeg, cv::IMREAD_REDUCED_COLOR_2);
            const auto t1 = std::chrono::steady_clock::now();
            if (decoded.empty()) {
                return out;
            }
            cpu_best = std::min(cpu_best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        out.race_won = gpu_best <= cpu_best * 0.8;
        LOG(INFO) << "gpu jpeg probe: backend="
                  << (out.backend == NVJPEG_BACKEND_HARDWARE ? "hw-nvjpeg" : "sm-nvjpeg") << " gpu=" << gpu_best
                  << "ms cpu=" << cpu_best << "ms race_" << (out.race_won ? "won" : "lost");
        return out;
    }();
    return result;
}

NvJpegPlan& decoder_plan() {
    static NvJpegPlan plan;
    if (!plan.ok && probe_once().capable) {
        if (build_plan(probe_once().backend, &plan)) {
            LOG(INFO) << "gpu jpeg decoder ready: " << backend_name();
        } else {
            release_plan(&plan);
        }
    }
    return plan;
}

std::mutex& decoder_mutex() {
    static std::mutex mu;
    return mu;
}

}  // namespace

bool recommended() {
    return probe_once().capable && probe_once().race_won;
}

bool available() {
    return probe_once().capable;
}

const char* backend_name() {
    const ProbeResult& probe = probe_once();
    if (!probe.capable) {
        return "unavailable";
    }
    if (probe.backend == NVJPEG_BACKEND_HARDWARE) {
        return "hw-nvjpeg";
    }
    return probe.race_won ? "sm-nvjpeg" : "sm-nvjpeg(slow,race-lost)";
}

cv::Mat decode(const unsigned char* data, size_t size, std::string* err) {
    if (err != nullptr) {
        err->clear();
    }
    if (!probe_once().capable) {
        if (err != nullptr) {
            *err = "gpu jpeg decoder unavailable";
        }
        return {};
    }
    const std::lock_guard<std::mutex> guard(decoder_mutex());
    NvJpegPlan& plan = decoder_plan();
    if (!plan.ok) {
        if (err != nullptr) {
            *err = "gpu jpeg decoder plan not ready";
        }
        return {};
    }
    int component_count = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};
    if (nvjpegGetImageInfo(plan.handle, data, size, &component_count, &subsampling, widths, heights) !=
        NVJPEG_STATUS_SUCCESS) {
        if (err != nullptr) {
            *err = "nvjpegGetImageInfo failed";
        }
        return {};
    }
    if (component_count != 3) {
        if (err != nullptr) {
            *err = "gpu path takes 3-component jpegs only";
        }
        return {};
    }
    const int out_w = widths[0];
    const int out_h = heights[0];
    if (out_w <= 0 || out_h <= 0 || out_w > 16384 || out_h > 16384) {
        if (err != nullptr) {
            *err = "unsupported decode dimensions";
        }
        return {};
    }
    const size_t out_bytes = static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3;
    if (!run_decode(&plan, data, size, out_w, out_h)) {
        if (err != nullptr) {
            *err = "nvjpegDecode failed";
        }
        return {};
    }
    if (plan.pinned_buffer_bytes < out_bytes) {
        if (plan.pinned_buffer != nullptr) {
            cudaFreeHost(plan.pinned_buffer);
            plan.pinned_buffer = nullptr;
            plan.pinned_buffer_bytes = 0;
        }
        if (cudaHostAlloc(reinterpret_cast<void**>(&plan.pinned_buffer), out_bytes, cudaHostAllocDefault) !=
            cudaSuccess) {
            if (err != nullptr) {
                *err = "cudaHostAlloc for jpeg staging failed";
            }
            return {};
        }
        plan.pinned_buffer_bytes = out_bytes;
    }
    if (cudaMemcpyAsync(plan.pinned_buffer, plan.device_buffer, out_bytes, cudaMemcpyDeviceToHost, plan.stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(plan.stream) != cudaSuccess) {
        if (err != nullptr) {
            *err = "jpeg output D2H failed";
        }
        return {};
    }
    return cv::Mat(out_h, out_w, CV_8UC3, plan.pinned_buffer).clone();
}

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq
