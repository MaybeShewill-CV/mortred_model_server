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
#include <opencv2/imgproc.hpp>

#include "glog/logging.h"

#ifdef MORTRED_HAS_JPEGGPU
#include <jpeggpu/jpeggpu.h>
#endif

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {

// global counters (three-layer visibility)
std::atomic<uint64_t> g_request_count[BACKEND_COUNT] = {};

namespace {

std::mutex& decoder_mutex() {
    static std::mutex mu;
    return mu;
}

#ifdef MORTRED_HAS_JPEGGPU

// ---------------------------------------------------------------------------
// jpeggpu backend (Huffman self-synchronizing parallel decode)
// ---------------------------------------------------------------------------

struct JpegGpuPlan {
    jpeggpu_decoder_t decoder = nullptr;
    cudaStream_t stream = nullptr;
    void* d_tmp = nullptr;
    size_t d_tmp_size = 0;
    uint8_t* d_planes[JPEGGPU_MAX_COMP] = {};
    size_t d_plane_sizes[JPEGGPU_MAX_COMP] = {};
    uint8_t* h_planes[JPEGGPU_MAX_COMP] = {};
    size_t h_plane_sizes[JPEGGPU_MAX_COMP] = {};
    uint8_t* h_jpeg_pinned = nullptr;
    size_t h_jpeg_pinned_size = 0;
    bool ok = false;

    bool init() {
        if (jpeggpu_decoder_startup(&decoder) != JPEGGPU_SUCCESS) return false;
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        ok = true;
        return true;
    }

    void release() {
        if (decoder) jpeggpu_decoder_cleanup(decoder);
        if (stream) cudaStreamDestroy(stream);
        if (d_tmp) cudaFree(d_tmp);
        for (int c = 0; c < JPEGGPU_MAX_COMP; ++c) {
            if (d_planes[c]) cudaFree(d_planes[c]);
            if (h_planes[c]) cudaFreeHost(h_planes[c]);
        }
        if (h_jpeg_pinned) cudaFreeHost(h_jpeg_pinned);
        memset(this, 0, sizeof(*this));
        ok = false;
    }

    cv::Mat decode(const unsigned char* data, size_t size, std::string* err) {
        if (err) err->clear();
        if (!ok) {
            if (err) *err = "jpeggpu plan not ready";
            return {};
        }
        // pinned host buffer for JPEG data (recommended by jpeggpu docs)
        if (h_jpeg_pinned_size < size) {
            if (h_jpeg_pinned) cudaFreeHost(h_jpeg_pinned);
            if (cudaHostAlloc((void**)&h_jpeg_pinned, size, cudaHostAllocDefault) != cudaSuccess) {
                h_jpeg_pinned = nullptr; h_jpeg_pinned_size = 0;
                if (err) *err = "pinned alloc for jpeg bytes failed";
                return {};
            }
            h_jpeg_pinned_size = size;
        }
        memcpy(h_jpeg_pinned, data, size);

        // parse header (CPU-only, fast)
        struct jpeggpu_img_info info;
        if (jpeggpu_decoder_parse_header(decoder, &info, h_jpeg_pinned, size) != JPEGGPU_SUCCESS) {
            if (err) *err = "jpeggpu parse_header failed";
            return {};
        }
        if (info.num_components != 3) {
            if (err) *err = "jpeggpu takes 3-component jpegs only";
            return {};
        }

        // temp GPU buffer
        size_t tmp_size = 0;
        if (jpeggpu_decoder_get_buffer_size(decoder, &tmp_size) != JPEGGPU_SUCCESS) {
            if (err) *err = "jpeggpu get_buffer_size failed";
            return {};
        }
        if (d_tmp_size < tmp_size) {
            if (d_tmp) cudaFree(d_tmp);
            if (cudaMalloc(&d_tmp, ((tmp_size + 255) / 256) * 256) != cudaSuccess) {
                d_tmp = nullptr; d_tmp_size = 0;
                if (err) *err = "temp device alloc failed";
                return {};
            }
            d_tmp_size = tmp_size;
        }

        // H2D transfer of JPEG data
        if (jpeggpu_decoder_transfer(decoder, d_tmp, d_tmp_size, stream) != JPEGGPU_SUCCESS) {
            if (err) *err = "jpeggpu transfer failed";
            return {};
        }

        // allocate/reuse output planes
        struct jpeggpu_img img = {};
        for (int c = 0; c < info.num_components; ++c) {
            size_t plane_bytes = (size_t)info.sizes_y[c] * info.sizes_x[c];
            if (d_plane_sizes[c] < plane_bytes) {
                if (d_planes[c]) cudaFree(d_planes[c]);
                if (h_planes[c]) cudaFreeHost(h_planes[c]);
                if (cudaMalloc((void**)&d_planes[c], plane_bytes) != cudaSuccess ||
                    cudaHostAlloc((void**)&h_planes[c], plane_bytes, cudaHostAllocDefault) != cudaSuccess) {
                    if (err) *err = "plane alloc failed";
                    return {};
                }
                d_plane_sizes[c] = plane_bytes;
                h_plane_sizes[c] = plane_bytes;
            }
            img.image[c] = d_planes[c];
            img.pitch[c] = ((info.sizes_x[c] + 7) / 8) * 8;
        }

        // GPU decode
        if (jpeggpu_decoder_decode(decoder, &img, d_tmp, d_tmp_size, stream) != JPEGGPU_SUCCESS) {
            if (err) *err = "jpeggpu decode failed";
            return {};
        }

        // D2H planes
        for (int c = 0; c < info.num_components; ++c) {
            size_t copy_bytes = (size_t)info.sizes_y[c] * info.sizes_x[c];
            cudaMemcpyAsync(h_planes[c], d_planes[c], copy_bytes, cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        // single-pass planar YCbCr → interleaved BGR (no intermediate Mats)
        cv::Mat y_plane(info.sizes_y[0], info.sizes_x[0], CV_8UC1, h_planes[0]);
        cv::Mat cb_plane, cr_plane;
        if (info.sizes_y[1] != info.sizes_y[0] || info.sizes_x[1] != info.sizes_x[0]) {
            cb_plane = cv::Mat(info.sizes_y[1], info.sizes_x[1], CV_8UC1, h_planes[1]);
            cr_plane = cv::Mat(info.sizes_y[2], info.sizes_x[2], CV_8UC1, h_planes[2]);
            cv::resize(cb_plane, cb_plane, y_plane.size(), 0, 0, cv::INTER_LINEAR);
            cv::resize(cr_plane, cr_plane, y_plane.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            cb_plane = cv::Mat(info.sizes_y[1], info.sizes_x[1], CV_8UC1, h_planes[1]);
            cr_plane = cv::Mat(info.sizes_y[2], info.sizes_x[2], CV_8UC1, h_planes[2]);
        }
        cv::Mat bgr(y_plane.rows, y_plane.cols, CV_8UC3);
        for (int y = 0; y < bgr.rows; ++y) {
            const uint8_t* y_src = y_plane.ptr<uint8_t>(y);
            const uint8_t* cb_src = cb_plane.ptr<uint8_t>(y);
            const uint8_t* cr_src = cr_plane.ptr<uint8_t>(y);
            uint8_t* dst = bgr.ptr<uint8_t>(y);
            for (int x = 0; x < bgr.cols; ++x) {
                const float yv = (float)y_src[x];
                const float cb = (float)cb_src[x] - 128.0f;
                const float cr = (float)cr_src[x] - 128.0f;
                dst[x*3+0] = (uint8_t)std::max(0.0f, std::min(255.0f, yv + 1.772f * cb));
                dst[x*3+1] = (uint8_t)std::max(0.0f, std::min(255.0f, yv - 0.344136f * cb - 0.714136f * cr));
                dst[x*3+2] = (uint8_t)std::max(0.0f, std::min(255.0f, yv + 1.402f * cr));
            }
        }
        return bgr;
    }
};

bool probe_jpeggpu_backend(const std::vector<unsigned char>& jpeg) {
    JpegGpuPlan plan;
    if (!plan.init()) {
        plan.release();
        return false;
    }
    cv::Mat result = plan.decode(jpeg.data(), jpeg.size(), nullptr);
    plan.release();
    return !result.empty();
}

double race_jpeggpu(JpegGpuPlan* plan, const std::vector<unsigned char>& jpeg, int rounds) {
    double best = 1e9;
    for (int i = 0; i < rounds; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (plan->decode(jpeg.data(), jpeg.size(), nullptr).empty()) return 1e9;
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

#endif // MORTRED_HAS_JPEGGPU

// ---------------------------------------------------------------------------
// nvjpeg backends (existing S1 infrastructure)
// ---------------------------------------------------------------------------

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

bool build_nvjpeg_plan(nvjpegBackend_t backend, NvJpegPlan* plan) {
    if (nvjpegCreateEx(backend, nullptr, nullptr, 0, &plan->handle) != NVJPEG_STATUS_SUCCESS ||
        nvjpegJpegStateCreate(plan->handle, &plan->state) != NVJPEG_STATUS_SUCCESS ||
        cudaStreamCreate(&plan->stream) != cudaSuccess) {
        return false;
    }
    plan->ok = true;
    return true;
}

void release_nvjpeg_plan(NvJpegPlan* plan) {
    if (plan->state) nvjpegJpegStateDestroy(plan->state);
    if (plan->handle) nvjpegDestroy(plan->handle);
    if (plan->stream) cudaStreamDestroy(plan->stream);
    if (plan->device_buffer) cudaFree(plan->device_buffer);
    if (plan->pinned_buffer) cudaFreeHost(plan->pinned_buffer);
    memset(plan, 0, sizeof(*plan));
    plan->ok = false;
}

bool run_nvjpeg_decode(NvJpegPlan* plan, const unsigned char* data, size_t size, int out_w, int out_h) {
    const size_t out_bytes = (size_t)out_w * (size_t)out_h * 3;
    if (plan->device_buffer_bytes < out_bytes) {
        if (plan->device_buffer) cudaFree(plan->device_buffer);
        if (cudaMalloc((void**)&plan->device_buffer, out_bytes) != cudaSuccess) return false;
        plan->device_buffer_bytes = out_bytes;
    }
    nvjpegImage_t image{};
    image.channel[0] = plan->device_buffer;
    image.pitch[0] = (size_t)out_w * 3;
    if (nvjpegDecode(plan->handle, plan->state, data, size, NVJPEG_OUTPUT_BGRI, &image, plan->stream) != NVJPEG_STATUS_SUCCESS)
        return false;
    return cudaStreamSynchronize(plan->stream) == cudaSuccess;
}

cv::Mat decode_nvjpeg(NvJpegPlan* plan, const unsigned char* data, size_t size, std::string* err) {
    int nComp; nvjpegChromaSubsampling_t css;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    if (nvjpegGetImageInfo(plan->handle, data, size, &nComp, &css, widths, heights) != NVJPEG_STATUS_SUCCESS) {
        if (err) *err = "nvjpegGetImageInfo failed";
        return {};
    }
    if (nComp != 3) {
        if (err) *err = "nvjpeg path takes 3-component jpegs only";
        return {};
    }
    const int w = widths[0], h = heights[0];
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384) {
        if (err) *err = "unsupported dimensions";
        return {};
    }
    const size_t bytes = (size_t)w * h * 3;
    if (!run_nvjpeg_decode(plan, data, size, w, h)) {
        if (err) *err = "nvjpegDecode failed";
        return {};
    }
    if (plan->pinned_buffer_bytes < bytes) {
        if (plan->pinned_buffer) cudaFreeHost(plan->pinned_buffer);
        if (cudaHostAlloc((void**)&plan->pinned_buffer, bytes, cudaHostAllocDefault) != cudaSuccess) {
            if (err) *err = "staging alloc failed";
            return {};
        }
        plan->pinned_buffer_bytes = bytes;
    }
    if (cudaMemcpyAsync(plan->pinned_buffer, plan->device_buffer, bytes, cudaMemcpyDeviceToHost, plan->stream) != cudaSuccess ||
        cudaStreamSynchronize(plan->stream) != cudaSuccess) {
        if (err) *err = "D2H failed";
        return {};
    }
    return cv::Mat(h, w, CV_8UC3, plan->pinned_buffer).clone();
}

// ---------------------------------------------------------------------------
// multi-image race (3 test images, majority verdict)
// ---------------------------------------------------------------------------

struct RaceImage {
    int width, height, quality;
    double noise;
    const char* label;
};

cv::Mat make_race_source(const RaceImage& spec) {
    cv::Mat img(spec.height, spec.width, CV_8UC3);
    cv::RNG rng(20260920);
    for (int y = 0; y < spec.height; ++y) {
        auto* row = img.ptr<unsigned char>(y);
        for (int x = 0; x < spec.width; ++x) {
            row[x*3+0] = (unsigned char)((x * 255) / spec.width);
            row[x*3+1] = (unsigned char)((y * 255) / spec.height);
            row[x*3+2] = (unsigned char)(((x + y) * 127) / spec.width);
        }
    }
    int amp = (int)(spec.noise * 28);
    cv::Mat noise(img.size(), CV_8UC3);
    rng.fill(noise, cv::RNG::UNIFORM, cv::Scalar(-amp,-amp,-amp), cv::Scalar(amp,amp,amp));
    cv::add(img, noise, img, cv::noArray(), CV_8UC3);
    return img;
}

std::vector<unsigned char> make_race_jpeg(const RaceImage& spec) {
    cv::Mat img = make_race_source(spec);
    std::vector<unsigned char> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, spec.quality};
    cv::imencode(".jpg", img, jpeg, params);
    return jpeg;
}

double race_cpu(const std::vector<unsigned char>& jpeg, int rounds) {
    double best = 1e9;
    for (int i = 0; i < rounds; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat decoded = cv::imdecode(jpeg, cv::IMREAD_REDUCED_COLOR_2);
        auto t1 = std::chrono::steady_clock::now();
        if (decoded.empty()) return 1e9;
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

// ---------------------------------------------------------------------------
// probe: capability ladder + multi-image race
// ---------------------------------------------------------------------------

struct ProbeResult {
    bool capable = false;
    bool race_won = false;
    Backend selected = CPU_REDUCED;
    // race detail for logging
    double race_gpu_ms[3] = {};
    double race_cpu_ms[3] = {};
};

const ProbeResult& probe_once() {
    static const ProbeResult result = [] {
        ProbeResult out;
        const int rounds = 3;
        static const RaceImage race_specs[] = {
            {640, 480, 50, 0.3, "small-fast"},
            {1024, 1024, 90, 1.0, "medium-typical"},
            {1920, 1080, 90, 1.0, "large-phone"},
        };
        std::vector<std::vector<unsigned char>> jpegs;
        for (const auto& spec : race_specs) {
            jpegs.push_back(make_race_jpeg(spec));
        }

#ifdef MORTRED_HAS_JPEGGPU
        // --- ladder level 1: jpeggpu ---
        {
            JpegGpuPlan plan;
            if (plan.init()) {
                cv::Mat test = plan.decode(jpegs[1].data(), jpegs[1].size(), nullptr);
                if (!test.empty()) {
                    // multi-image race: GPU must win ≥2/3 images by ≥20%
                    int wins = 0;
                    for (int i = 0; i < 3; ++i) {
                        out.race_gpu_ms[i] = race_jpeggpu(&plan, jpegs[i], rounds);
                        out.race_cpu_ms[i] = race_cpu(jpegs[i], rounds);
                        if (out.race_gpu_ms[i] <= out.race_cpu_ms[i] * 0.8) ++wins;
                    }
                    out.capable = true;
                    out.selected = JPEGGPU;
                    out.race_won = wins >= 2;
                    LOG(INFO) << "gpu jpeg probe: backend=jpeggpu"
                              << " wins=" << wins << "/3"
                              << " gpu_ms=[" << out.race_gpu_ms[0] << "," << out.race_gpu_ms[1] << "," << out.race_gpu_ms[2] << "]"
                              << " cpu_ms=[" << out.race_cpu_ms[0] << "," << out.race_cpu_ms[1] << "," << out.race_cpu_ms[2] << "]"
                              << " race_" << (out.race_won ? "won" : "lost");
                    plan.release();
                    if (out.race_won || out.capable) return out;  // jpeggpu is the best backend; race decides auto
                }
                plan.release();
            }
        }
#endif // MORTRED_HAS_JPEGGPU

        // --- ladder level 2-3: nvjpeg ---
        {
            NvJpegPlan probe_plan;
            bool have = false;
            nvjpegBackend_t chosen = NVJPEG_BACKEND_GPU_HYBRID;
            for (const nvjpegBackend_t candidate : {NVJPEG_BACKEND_HARDWARE, NVJPEG_BACKEND_GPU_HYBRID}) {
                if (!build_nvjpeg_plan(candidate, &probe_plan)) {
                    release_nvjpeg_plan(&probe_plan);
                    continue;
                }
                if (run_nvjpeg_decode(&probe_plan, jpegs[1].data(), jpegs[1].size(), 1024, 1024)) {
                    chosen = candidate;
                    have = true;
                    break;
                }
                release_nvjpeg_plan(&probe_plan);
            }
            if (!have) return out;

            // multi-image race
            int wins = 0;
            for (int i = 0; i < 3; ++i) {
                double gpu_best = 1e9;
                for (int r = 0; r < rounds; ++r) {
                    auto t0 = std::chrono::steady_clock::now();
                    int nComp; nvjpegChromaSubsampling_t css;
                    int w[NVJPEG_MAX_COMPONENT], h[NVJPEG_MAX_COMPONENT];
                    nvjpegGetImageInfo(probe_plan.handle, jpegs[i].data(), jpegs[i].size(), &nComp, &css, w, h);
                    run_nvjpeg_decode(&probe_plan, jpegs[i].data(), jpegs[i].size(), w[0], h[0]);
                    auto t1 = std::chrono::steady_clock::now();
                    gpu_best = std::min(gpu_best, std::chrono::duration<double, std::milli>(t1 - t0).count());
                }
                out.race_gpu_ms[i] = gpu_best;
                out.race_cpu_ms[i] = race_cpu(jpegs[i], rounds);
                if (gpu_best <= out.race_cpu_ms[i] * 0.8) ++wins;
            }
            release_nvjpeg_plan(&probe_plan);
            out.capable = true;
            out.selected = chosen == NVJPEG_BACKEND_HARDWARE ? NVJPEG_HW : NVJPEG_SM;
            out.race_won = wins >= 2;
            LOG(INFO) << "gpu jpeg probe: backend=" << (chosen == NVJPEG_BACKEND_HARDWARE ? "nvjpeg-hw" : "nvjpeg-sm")
                      << " wins=" << wins << "/3"
                      << " race_" << (out.race_won ? "won" : "lost");
        }
        return out;
    }();
    return result;
}

// ---------------------------------------------------------------------------
// production decoder plans (lazy init)
// ---------------------------------------------------------------------------

#ifdef MORTRED_HAS_JPEGGPU
JpegGpuPlan& jpeggpu_plan() {
    static JpegGpuPlan plan;
    if (!plan.ok && probe_once().capable && probe_once().selected == JPEGGPU) {
        if (plan.init()) {
            LOG(INFO) << "jpeggpu decoder ready";
        } else {
            plan.release();
        }
    }
    return plan;
}
#endif

NvJpegPlan& nvjpeg_plan() {
    static NvJpegPlan plan;
    if (!plan.ok && probe_once().capable && probe_once().selected >= NVJPEG_HW) {
        const nvjpegBackend_t backend =
            probe_once().selected == NVJPEG_HW ? NVJPEG_BACKEND_HARDWARE : NVJPEG_BACKEND_GPU_HYBRID;
        if (build_nvjpeg_plan(backend, &plan)) {
            LOG(INFO) << "nvjpeg decoder ready: " << backend_name();
        } else {
            release_nvjpeg_plan(&plan);
        }
    }
    return plan;
}

}  // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

bool recommended() {
    return probe_once().capable && probe_once().race_won;
}

bool available() {
    return probe_once().capable;
}

const char* backend_name() {
    const ProbeResult& p = probe_once();
    if (!p.capable) return "unavailable";
    switch (p.selected) {
        case JPEGGPU: return "jpeggpu";
        case NVJPEG_HW: return "nvjpeg-hw";
        case NVJPEG_SM: return "nvjpeg-sm";
        default: return "unavailable";
    }
}

const char* selected_backend_name() {
    return backend_name();
}

cv::Mat decode(const unsigned char* data, size_t size, std::string* err) {
    if (err) err->clear();
    if (!probe_once().capable) {
        if (err) *err = "gpu jpeg decoder unavailable";
        return {};
    }
    const std::lock_guard<std::mutex> guard(decoder_mutex());
    const Backend selected = probe_once().selected;

#ifdef MORTRED_HAS_JPEGGPU
    if (selected == JPEGGPU) {
        JpegGpuPlan& plan = jpeggpu_plan();
        if (!plan.ok) {
            if (err) *err = "jpeggpu plan not ready";
            g_request_count[FALLBACK].fetch_add(1);
            return {};
        }
        cv::Mat result = plan.decode(data, size, err);
        if (!result.empty()) {
            g_request_count[JPEGGPU].fetch_add(1);
        } else {
            g_request_count[FALLBACK].fetch_add(1);
        }
        return result;
    }
#endif

    // nvjpeg path
    NvJpegPlan& plan = nvjpeg_plan();
    if (!plan.ok) {
        if (err) *err = "nvjpeg plan not ready";
        g_request_count[FALLBACK].fetch_add(1);
        return {};
    }
    cv::Mat result = decode_nvjpeg(&plan, data, size, err);
    if (!result.empty()) {
        g_request_count[selected].fetch_add(1);
    } else {
        g_request_count[FALLBACK].fetch_add(1);
    }
    return result;
}

PlanarImage decode_planar(const unsigned char* data, size_t size, std::string* err) {
    PlanarImage out;
    if (err) err->clear();
#ifdef MORTRED_HAS_JPEGGPU
    if (!probe_once().capable || probe_once().selected != JPEGGPU) {
        if (err) *err = "planar decode requires jpeggpu backend";
        return out;
    }
    const std::lock_guard<std::mutex> guard(decoder_mutex());
    JpegGpuPlan& plan = jpeggpu_plan();
    if (!plan.ok) {
        if (err) *err = "jpeggpu plan not ready";
        return out;
    }
    // reuse the decode logic but return planar instead of converting to BGR
    // (this is the same code as JpegGpuPlan::decode up to the D2H step)
    if (!plan.ok) {
        if (err) *err = "plan not ready";
        return out;
    }
    if (plan.h_jpeg_pinned_size < size) {
        if (plan.h_jpeg_pinned) cudaFreeHost(plan.h_jpeg_pinned);
        if (cudaHostAlloc((void**)&plan.h_jpeg_pinned, size, cudaHostAllocDefault) != cudaSuccess) {
            plan.h_jpeg_pinned = nullptr; plan.h_jpeg_pinned_size = 0;
            if (err) *err = "pinned alloc failed";
            return out;
        }
        plan.h_jpeg_pinned_size = size;
    }
    memcpy(plan.h_jpeg_pinned, data, size);
    struct jpeggpu_img_info info;
    if (jpeggpu_decoder_parse_header(plan.decoder, &info, plan.h_jpeg_pinned, size) != JPEGGPU_SUCCESS) {
        if (err) *err = "parse_header failed";
        return out;
    }
    if (info.num_components != 3) {
        if (err) *err = "3-component only";
        return out;
    }
    size_t tmp_size = 0;
    if (jpeggpu_decoder_get_buffer_size(plan.decoder, &tmp_size) != JPEGGPU_SUCCESS) {
        if (err) *err = "get_buffer_size failed";
        return out;
    }
    if (plan.d_tmp_size < tmp_size) {
        if (plan.d_tmp) cudaFree(plan.d_tmp);
        if (cudaMalloc(&plan.d_tmp, ((tmp_size + 255) / 256) * 256) != cudaSuccess) {
            plan.d_tmp = nullptr; plan.d_tmp_size = 0;
            if (err) *err = "temp alloc failed";
            return out;
        }
        plan.d_tmp_size = tmp_size;
    }
    if (jpeggpu_decoder_transfer(plan.decoder, plan.d_tmp, plan.d_tmp_size, plan.stream) != JPEGGPU_SUCCESS) {
        if (err) *err = "transfer failed";
        return out;
    }
    struct jpeggpu_img img = {};
    for (int c = 0; c < info.num_components; ++c) {
        size_t plane_bytes = (size_t)info.sizes_y[c] * info.sizes_x[c];
        if (plan.d_plane_sizes[c] < plane_bytes) {
            if (plan.d_planes[c]) cudaFree(plan.d_planes[c]);
            if (plan.h_planes[c]) cudaFreeHost(plan.h_planes[c]);
            if (cudaMalloc((void**)&plan.d_planes[c], plane_bytes) != cudaSuccess ||
                cudaHostAlloc((void**)&plan.h_planes[c], plane_bytes, cudaHostAllocDefault) != cudaSuccess) {
                if (err) *err = "plane alloc failed";
                return out;
            }
            plan.d_plane_sizes[c] = plane_bytes;
            plan.h_plane_sizes[c] = plane_bytes;
        }
        img.image[c] = plan.d_planes[c];
        img.pitch[c] = ((info.sizes_x[c] + 7) / 8) * 8;
    }
    if (jpeggpu_decoder_decode(plan.decoder, &img, plan.d_tmp, plan.d_tmp_size, plan.stream) != JPEGGPU_SUCCESS) {
        if (err) *err = "decode failed";
        return out;
    }
    for (int c = 0; c < info.num_components; ++c) {
        size_t copy_bytes = (size_t)info.sizes_y[c] * info.sizes_x[c];
        cudaMemcpyAsync(plan.h_planes[c], plan.d_planes[c], copy_bytes, cudaMemcpyDeviceToHost, plan.stream);
    }
    cudaStreamSynchronize(plan.stream);
    // return planar without BGR conversion
    out.y  = cv::Mat(info.sizes_y[0], info.sizes_x[0], CV_8UC1, plan.h_planes[0]).clone();
    out.cb = cv::Mat(info.sizes_y[1], info.sizes_x[1], CV_8UC1, plan.h_planes[1]).clone();
    out.cr = cv::Mat(info.sizes_y[2], info.sizes_x[2], CV_8UC1, plan.h_planes[2]).clone();
    g_request_count[JPEGGPU].fetch_add(1);
#endif
    return out;
}

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq
