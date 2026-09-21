/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder.cpp
 * Date: 26-9-20
 ************************************************/

#include "models/backend/gpu_jpeg_decoder.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

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

/*** Pool for zero-copy OUTPUT buffers. The old code reused one static
 * buffer per size class — under concurrent workers request B's preprocess
 * kernel could overwrite the tensor request A's TRT inference was still
 * reading (different streams, no ordering). Buffers now leave the pool only
 * while in flight and come back via release_pipeline_buffers(), which the
 * caller invokes only after the consumer session has synchronized. The
 * freelist is capped; excess buffers are cudaFree'd. */
struct OutputBufferPool {
    static constexpr size_t k_max_free = 16;
    std::mutex mu;
    std::deque<void*> ptrs;
    std::deque<size_t> sizes;

    void* acquire(size_t bytes) {
        {
            const std::lock_guard<std::mutex> guard(mu);
            for (size_t i = 0; i < ptrs.size(); ++i) {
                if (sizes[i] >= bytes) {
                    void* p = ptrs[i];
                    ptrs.erase(ptrs.begin() + static_cast<long>(i));
                    sizes.erase(sizes.begin() + static_cast<long>(i));
                    return p;
                }
            }
        }
        void* p = nullptr;
        return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
    }

    void give_back(void* p, size_t bytes) {
        if (p == nullptr) return;
        const std::lock_guard<std::mutex> guard(mu);
        if (ptrs.size() >= k_max_free) {
            cudaFree(p);
            return;
        }
        ptrs.push_back(p);
        sizes.push_back(bytes);
    }
};

OutputBufferPool& output_pool() {
    static OutputBufferPool pool;
    return pool;
}

#ifdef MORTRED_HAS_JPEGGPU

// ---------------------------------------------------------------------------
// jpeggpu backend (Huffman self-synchronizing parallel decode)
// ---------------------------------------------------------------------------

struct JpegGpuPlan {
    jpeggpu_decoder_t decoder = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t frame_ready_event = nullptr;  // recorded after each preprocess launch
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
        if (cudaEventCreateWithFlags(&frame_ready_event, cudaEventDisableTiming) != cudaSuccess) return false;
        ok = true;
        return true;
    }

    void release() {
        if (decoder) jpeggpu_decoder_cleanup(decoder);
        if (stream) cudaStreamDestroy(stream);
        if (frame_ready_event) cudaEventDestroy(frame_ready_event);
        if (d_tmp) cudaFree(d_tmp);
        for (int c = 0; c < JPEGGPU_MAX_COMP; ++c) {
            if (d_planes[c]) cudaFree(d_planes[c]);
            if (h_planes[c]) cudaFreeHost(h_planes[c]);
        }
        if (h_jpeg_pinned) cudaFreeHost(h_jpeg_pinned);
        memset(this, 0, sizeof(*this));
        ok = false;
    }
};

#endif // MORTRED_HAS_JPEGGPU

// ---------------------------------------------------------------------------
// capability probe: jpeggpu compiles, inits, and decodes one small JPEG.
// No timing, no race — path choice is a static threshold decision made per
// request by the caller (BackendCvModel's decode fork).
// ---------------------------------------------------------------------------

struct ProbeResult {
    bool capable = false;
    Backend selected = CPU_REDUCED;
};

#ifdef MORTRED_HAS_JPEGGPU
std::vector<unsigned char> make_probe_jpeg() {
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(96, 128, 160));
    std::vector<unsigned char> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", img, jpeg, params);
    return jpeg;
}

bool probe_jpeggpu_capability() {
    JpegGpuPlan plan;
    if (!plan.init()) {
        plan.release();
        return false;
    }
    const std::vector<unsigned char> jpeg = make_probe_jpeg();
    if (plan.h_jpeg_pinned_size < jpeg.size()) {
        if (plan.h_jpeg_pinned) cudaFreeHost(plan.h_jpeg_pinned);
        if (cudaHostAlloc((void**)&plan.h_jpeg_pinned, jpeg.size(), cudaHostAllocDefault) != cudaSuccess) {
            plan.h_jpeg_pinned = nullptr; plan.h_jpeg_pinned_size = 0;
            plan.release();
            return false;
        }
        plan.h_jpeg_pinned_size = jpeg.size();
    }
    std::memcpy(plan.h_jpeg_pinned, jpeg.data(), jpeg.size());
    struct jpeggpu_img_info info;
    if (jpeggpu_decoder_parse_header(plan.decoder, &info, plan.h_jpeg_pinned, jpeg.size()) != JPEGGPU_SUCCESS) {
        plan.release();
        return false;
    }
    size_t tmp_size = 0;
    if (jpeggpu_decoder_get_buffer_size(plan.decoder, &tmp_size) != JPEGGPU_SUCCESS) {
        plan.release();
        return false;
    }
    if (plan.d_tmp_size < tmp_size) {
        if (plan.d_tmp) cudaFree(plan.d_tmp);
        if (cudaMalloc(&plan.d_tmp, ((tmp_size + 255) / 256) * 256) != cudaSuccess) {
            plan.d_tmp = nullptr; plan.d_tmp_size = 0;
            plan.release();
            return false;
        }
        plan.d_tmp_size = ((tmp_size + 255) / 256) * 256;
    }
    if (jpeggpu_decoder_transfer(plan.decoder, plan.d_tmp, plan.d_tmp_size, plan.stream) == JPEGGPU_SUCCESS) {
        struct jpeggpu_img img = {};
        uint8_t* planes[JPEGGPU_MAX_COMP] = {};
        for (int c = 0; c < info.num_components; ++c) {
            img.pitch[c] = ((info.sizes_x[c] + 7) / 8) * 8;
            size_t bytes = (size_t)info.sizes_y[c] * img.pitch[c];
            if (cudaMalloc(&planes[c], bytes) != cudaSuccess) break;
            img.image[c] = planes[c];
        }
        const bool ok = img.image[info.num_components - 1] != nullptr &&
                        jpeggpu_decoder_decode(plan.decoder, &img, plan.d_tmp, plan.d_tmp_size,
                                               plan.stream) == JPEGGPU_SUCCESS &&
                        cudaStreamSynchronize(plan.stream) == cudaSuccess;
        for (int c = 0; c < JPEGGPU_MAX_COMP; ++c) {
            if (planes[c]) cudaFree(planes[c]);
        }
        plan.release();
        return ok;
    }
    plan.release();
    return false;
}
#endif

const ProbeResult& probe_once() {
    static const ProbeResult result = [] {
        ProbeResult out;
#ifdef MORTRED_HAS_JPEGGPU
        if (probe_jpeggpu_capability()) {
            out.capable = true;
            out.selected = JPEGGPU;
            LOG(INFO) << "gpu jpeg probe: backend=jpeggpu capable=yes (no race - static fork thresholds decide the path)";
        } else {
            LOG(INFO) << "gpu jpeg probe: jpeggpu not capable, gpu decode path disabled";
        }
#endif
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

}  // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

const char* backend_name() {
    return probe_once().capable ? "jpeggpu" : "unavailable";
}

const char* selected_backend_name() {
    return backend_name();
}

std::string render_decode_metrics() {
    static const char* const k_names[BACKEND_COUNT] = {
        "jpeggpu", "nvjpeg-hw", "nvjpeg-sm", "cpu-reduced", "cpu-full", "fallback"};
    const ProbeResult& p = probe_once();
    std::ostringstream ss;
    ss << "# HELP mortred_jpeg_decode_total Images decoded per backend\n";
    ss << "# TYPE mortred_jpeg_decode_total counter\n";
    for (int b = 0; b < BACKEND_COUNT; ++b) {
        const uint64_t n = g_request_count[b].load();
        if (n == 0) {
            continue;
        }
        ss << "mortred_jpeg_decode_total{backend=\"" << k_names[b] << "\"} " << n << "\n";
    }
    ss << "# HELP mortred_jpeg_decode_ladder Decode backend capability state\n";
    ss << "# TYPE mortred_jpeg_decode_ladder gauge\n";
    ss << "mortred_jpeg_decode_ladder{backend=\"" << backend_name()
       << "\",capable=\"" << (p.capable ? "yes" : "no")
       << "\"} 1\n";
    return ss.str();
}

DevicePlanes decode_to_device(const unsigned char* data, size_t size, std::string* err) {
    DevicePlanes out;
    if (err) err->clear();
#ifdef MORTRED_HAS_JPEGGPU
    if (!probe_once().capable || probe_once().selected != JPEGGPU) {
        if (err) *err = "device decode requires jpeggpu backend";
        return out;
    }
    const std::lock_guard<std::mutex> guard(decoder_mutex());
    JpegGpuPlan& plan = jpeggpu_plan();
    if (!plan.ok) {
        if (err) *err = "jpeggpu plan not ready";
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
        // jpeggpu writes full uint2 row segments at an 8-aligned pitch:
        // allocate stride*height, never packed width*height
        img.pitch[c] = ((info.sizes_x[c] + 7) / 8) * 8;
        size_t plane_bytes = (size_t)info.sizes_y[c] * img.pitch[c];
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
    }
    if (jpeggpu_decoder_decode(plan.decoder, &img, plan.d_tmp, plan.d_tmp_size, plan.stream) != JPEGGPU_SUCCESS) {
        if (err) *err = "decode failed";
        return out;
    }
    // NO cudaStreamSynchronize here: GPU work is submitted asynchronously.
    // The decode timing mark fires after submission (CPU returns immediately),
    // and the actual GPU completion is awaited in fetch_from_device() when
    // the decoded data is needed. This pipelines CPU and GPU work.
    // return device pointers — NO D2H, NO cv::Mat wrapping
    out.dev_y = plan.d_planes[0];
    out.dev_cb = plan.d_planes[1];
    out.dev_cr = plan.d_planes[2];
    out.y_w = info.sizes_x[0]; out.y_h = info.sizes_y[0];
    out.cb_w = info.sizes_x[1]; out.cb_h = info.sizes_y[1];
    out.y_stride = img.pitch[0];
    out.cb_stride = img.pitch[1];
    out.valid = true;
    g_request_count[JPEGGPU].fetch_add(1);
#endif
    return out;
}

// CUDA kernel launcher from the parameterized gpu_preprocess.cu
// (declared before all callers)
#ifdef MORTRED_HAS_JPEGGPU
extern "C" cudaError_t launch_preprocess(
    const uint8_t* d_y, const uint8_t* d_cb, const uint8_t* d_cr,
    int src_w, int src_h, int cb_w, int cb_h,
    int y_stride, int cb_stride,
    void* d_out, void* d_gray_out,
    int out_w, int out_h,
    int resize_type, int color_order, int rotation,
    float norm_scale,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2,
    float pad_val,
    int pad_zero_norm,
    int output_is_fp16,
    int output_is_nhwc,
    int unpad_w, int unpad_h, int pad_x, int pad_y,
    int crop_x, int crop_y,
    cudaStream_t stream);
#endif

GpuPipelineResult decode_and_preprocess(
    const unsigned char* data, size_t size,
    int network_w, int network_h,
    const GpuPreprocessDescriptor& desc,
    std::string* err) {

    GpuPipelineResult out;
    if (err) err->clear();
    if (!desc.valid()) {
        if (err) *err = "invalid GpuPreprocessDescriptor";
        return out;
    }
#ifdef MORTRED_HAS_JPEGGPU
    if (!probe_once().capable || probe_once().selected != JPEGGPU) {
        if (err) *err = "GPU zero-copy pipeline requires jpeggpu backend";
        return out;
    }

    // Step 1: decode JPEG to device (async)
    DevicePlanes dp = decode_to_device(data, size, err);
    if (!dp.valid) return out;

    // Step 2: compute geometry based on the descriptor's Resize type (CPU, fast)
    int unpad_w = 0, unpad_h = 0, pad_x = 0, pad_y = 0, crop_x = 0, crop_y = 0;
    int eff_out_w = network_w, eff_out_h = network_h;

    switch (desc.resize) {
    case GpuPreprocessDescriptor::Resize::LETTERBOX:
    case GpuPreprocessDescriptor::Resize::KEEP_RATIO_PAD_CENTER: {
        const double ratio = std::min(
            (double)network_h / (double)dp.y_h,
            (double)network_w / (double)dp.y_w);
        unpad_w = (int)std::round(dp.y_w * ratio);
        unpad_h = (int)std::round(dp.y_h * ratio);
        if (unpad_w > network_w) unpad_w = network_w;
        if (unpad_h > network_h) unpad_h = network_h;
        const double dw = ((double)network_w - unpad_w) / 2.0;
        const double dh = ((double)network_h - unpad_h) / 2.0;
        pad_x = std::max(0, (int)std::round(dw - 0.1));
        pad_y = std::max(0, (int)std::round(dh - 0.1));
        break;
    }
    case GpuPreprocessDescriptor::Resize::KEEP_RATIO_PAD_ZERO: {
        // DepthAnything-style: the resized image anchors at the top-left and
        // the zero pad lands on the right/bottom, matching the CPU path's
        // Mat::zeros + copyTo(Rect(0,0,...)) and its top-left crop in post
        const double ratio = std::min(
            (double)network_h / (double)dp.y_h,
            (double)network_w / (double)dp.y_w);
        unpad_w = (int)std::round(dp.y_w * ratio);
        unpad_h = (int)std::round(dp.y_h * ratio);
        if (unpad_w > network_w) unpad_w = network_w;
        if (unpad_h > network_h) unpad_h = network_h;
        pad_x = 0;
        pad_y = 0;
        break;
    }
    case GpuPreprocessDescriptor::Resize::CENTER_CROP: {
        // resize source to pre_crop_size, then crop center to network size
        const int pre_w = desc.pre_crop_size.width > 0 ? desc.pre_crop_size.width : network_w;
        const int pre_h = desc.pre_crop_size.height > 0 ? desc.pre_crop_size.height : network_h;
        crop_x = std::max(0, (pre_w - network_w) / 2);
        crop_y = std::max(0, (pre_h - network_h) / 2);
        unpad_w = pre_w;
        unpad_h = pre_h;
        break;
    }
    case GpuPreprocessDescriptor::Resize::DIRECT_RESIZE: {
        unpad_w = network_w;
        unpad_h = network_h;
        break;
    }
    case GpuPreprocessDescriptor::Resize::ALIGN_TO_MULTIPLE: {
        eff_out_w = ((dp.y_w + desc.align_multiple - 1) / desc.align_multiple) * desc.align_multiple;
        eff_out_h = ((dp.y_h + desc.align_multiple - 1) / desc.align_multiple) * desc.align_multiple;
        unpad_w = eff_out_w;
        unpad_h = eff_out_h;
        break;
    }
    case GpuPreprocessDescriptor::Resize::NONE: {
        eff_out_w = dp.y_w;
        eff_out_h = dp.y_h;
        unpad_w = eff_out_w;
        unpad_h = eff_out_h;
        break;
    }
    }

    // Step 3: acquire output buffers from the pool (NOT static reuse —
    // concurrent workers must never share an in-flight output buffer)
    const int channels = (desc.color == GpuPreprocessDescriptor::Color::GRAY) ? 1 : 3;
    const size_t elem_size = (desc.output_dtype == DType::F16) ? 2 : 4;
    const size_t out_bytes = (size_t)eff_out_h * eff_out_w * channels * elem_size;
    void* d_output = output_pool().acquire(out_bytes);
    if (d_output == nullptr) {
        if (err) *err = "output buffer alloc failed";
        return out;
    }

    // Optional gray secondary output
    void* d_gray = nullptr;
    size_t gray_bytes = 0;
    if (desc.secondary_gray_output) {
        gray_bytes = (size_t)eff_out_h * eff_out_w * elem_size;
        d_gray = output_pool().acquire(gray_bytes);
        if (d_gray == nullptr) {
            output_pool().give_back(d_output, out_bytes);
            if (err) *err = "gray output alloc failed";
            return out;
        }
    }

    // Step 4: launch parameterized CUDA kernel (async, same stream)
    const std::lock_guard<std::mutex> guard(decoder_mutex());
    JpegGpuPlan& plan = jpeggpu_plan();
    if (!plan.ok) {
        output_pool().give_back(d_output, out_bytes);
        if (d_gray) output_pool().give_back(d_gray, gray_bytes);
        if (err) *err = "plan not ready";
        return out;
    }
    const int resize_type = static_cast<int>(desc.resize);
    const int color_order = static_cast<int>(desc.color);
    const int rotation = static_cast<int>(desc.rotation);
    const float pad_val = (float)desc.pad_value;

    const cudaError_t launch_err = launch_preprocess(
        dp.dev_y, dp.dev_cb, dp.dev_cr,
        dp.y_w, dp.y_h, dp.cb_w, dp.cb_h,
        (int)(dp.y_stride > 0 ? dp.y_stride : (size_t)dp.y_w),
        (int)(dp.cb_stride > 0 ? dp.cb_stride : (size_t)dp.cb_w),
        d_output, d_gray,
        eff_out_w, eff_out_h,
        resize_type, color_order, rotation,
        desc.norm.scale,
        desc.norm.mean[0], desc.norm.mean[1], desc.norm.mean[2],
        desc.norm.std[0], desc.norm.std[1], desc.norm.std[2],
        pad_val,
        desc.pad_with_mean ? 1 : 0,
        desc.output_dtype == DType::F16 ? 1 : 0,
        desc.output_nhwc ? 1 : 0,
        unpad_w, unpad_h, pad_x, pad_y,
        crop_x, crop_y,
        plan.stream);
    if (launch_err != cudaSuccess) {
        output_pool().give_back(d_output, out_bytes);
        if (d_gray) output_pool().give_back(d_gray, gray_bytes);
        if (err) *err = std::string("CUDA kernel launch failed: ") + cudaGetErrorString(launch_err);
        return out;
    }

    // Order the consumer: everything on plan.stream up to here (H2D of the
    // jpeg bytes, decode, preprocess) must complete before another stream
    // reads the outputs. The consumer enqueues cudaStreamWaitEvent on this
    // event before its inference.
    if (cudaEventRecord(plan.frame_ready_event, plan.stream) != cudaSuccess) {
        output_pool().give_back(d_output, out_bytes);
        if (d_gray) output_pool().give_back(d_gray, gray_bytes);
        if (err) *err = "frame ready event record failed";
        return out;
    }

    out.device_input = d_output;
    out.device_gray = d_gray;
    out.out_w = eff_out_w;
    out.out_h = eff_out_h;
    out.src_w = dp.y_w;
    out.src_h = dp.y_h;
    out.ready_event = plan.frame_ready_event;
    out.device_input_bytes = out_bytes;
    out.device_gray_bytes = gray_bytes;
    out.valid = true;
    // decode_to_device already counted this decode
#endif
    return out;
}

void release_pipeline_buffers(const GpuPipelineResult& r) {
#ifdef MORTRED_HAS_JPEGGPU
    if (r.device_input != nullptr) {
        output_pool().give_back(r.device_input, r.device_input_bytes);
    }
    if (r.device_gray != nullptr) {
        output_pool().give_back(r.device_gray, r.device_gray_bytes);
    }
#endif
}

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq
