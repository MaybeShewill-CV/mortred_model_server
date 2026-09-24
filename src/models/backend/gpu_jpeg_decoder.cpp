/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder.cpp
 * Date: 26-9-20
 ************************************************/

#include "models/backend/gpu_jpeg_decoder.h"

#include <sstream>

#ifdef MORTRED_HAS_JPEGGPU
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <jpeggpu/jpeggpu.h>

#include "glog/logging.h"
#endif

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {

std::atomic<uint64_t> g_request_count[BACKEND_COUNT] = {};

namespace {

#ifdef MORTRED_HAS_JPEGGPU

std::atomic<int> g_open_slots{0};

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

bool grow_device(void** ptr, size_t* cap, size_t need) {
    if (need <= *cap && *ptr != nullptr) {
        return true;
    }
    void* fresh = nullptr;
    const size_t alloc = (need + 255u) & ~size_t{255};
    if (cudaMalloc(&fresh, alloc) != cudaSuccess) {
        return false;
    }
    if (*ptr != nullptr) {
        cudaFree(*ptr);
    }
    *ptr = fresh;
    *cap = alloc;
    return true;
}

bool grow_pinned(uint8_t** ptr, size_t* cap, size_t need) {
    if (need <= *cap && *ptr != nullptr) {
        return true;
    }
    uint8_t* fresh = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&fresh), need, cudaHostAllocDefault) != cudaSuccess) {
        return false;
    }
    if (*ptr != nullptr) {
        cudaFreeHost(*ptr);
    }
    *ptr = fresh;
    *cap = need;
    return true;
}

std::vector<unsigned char> make_probe_jpeg() {
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(96, 128, 160));
    std::vector<unsigned char> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", img, jpeg, params);
    return jpeg;
}

/*** Stack decoder, destroyed before return. Not the worker's slot. */
bool probe_jpeggpu() {
    jpeggpu_decoder_t decoder = nullptr;
    cudaStream_t stream = nullptr;
    uint8_t* pinned = nullptr;
    void* d_tmp = nullptr;
    uint8_t* planes[JPEGGPU_MAX_COMP] = {};
    bool ok = false;

    const std::vector<unsigned char> jpeg = make_probe_jpeg();
    if (jpeg.empty() || jpeggpu_decoder_startup(&decoder) != JPEGGPU_SUCCESS ||
        cudaStreamCreate(&stream) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&pinned), jpeg.size(), cudaHostAllocDefault) != cudaSuccess) {
        goto done;
    }
    std::memcpy(pinned, jpeg.data(), jpeg.size());
    {
        struct jpeggpu_img_info info;
        if (jpeggpu_decoder_parse_header(decoder, &info, pinned, jpeg.size()) != JPEGGPU_SUCCESS) {
            goto done;
        }
        size_t tmp_size = 0;
        if (jpeggpu_decoder_get_buffer_size(decoder, &tmp_size) != JPEGGPU_SUCCESS ||
            cudaMalloc(&d_tmp, (tmp_size + 255u) & ~size_t{255}) != cudaSuccess ||
            jpeggpu_decoder_transfer(decoder, d_tmp, tmp_size, stream) != JPEGGPU_SUCCESS) {
            goto done;
        }
        struct jpeggpu_img img = {};
        bool planes_ok = info.num_components > 0;
        for (int c = 0; c < info.num_components && c < JPEGGPU_MAX_COMP; ++c) {
            img.pitch[c] = ((info.sizes_x[c] + 7) / 8) * 8;
            const size_t bytes = static_cast<size_t>(info.sizes_y[c]) * img.pitch[c];
            if (cudaMalloc(reinterpret_cast<void**>(&planes[c]), bytes) != cudaSuccess) {
                planes_ok = false;
                break;
            }
            img.image[c] = planes[c];
        }
        ok = planes_ok &&
             jpeggpu_decoder_decode(decoder, &img, d_tmp, tmp_size, stream) == JPEGGPU_SUCCESS;
    }
done:
    if (stream != nullptr && cudaStreamSynchronize(stream) != cudaSuccess) {
        ok = false;
    }
    for (int c = 0; c < JPEGGPU_MAX_COMP; ++c) {
        if (planes[c] != nullptr) {
            cudaFree(planes[c]);
        }
    }
    if (d_tmp != nullptr) {
        cudaFree(d_tmp);
    }
    if (pinned != nullptr) {
        cudaFreeHost(pinned);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (decoder != nullptr) {
        jpeggpu_decoder_cleanup(decoder);
    }
    return ok;
}

struct Geometry {
    int unpad_w = 0;
    int unpad_h = 0;
    int pad_x = 0;
    int pad_y = 0;
    int crop_x = 0;
    int crop_y = 0;
    int out_w = 0;
    int out_h = 0;
};

Geometry compute_geometry(int src_w, int src_h, int network_w, int network_h,
                          const GpuPreprocessDescriptor& desc) {
    Geometry g;
    g.out_w = network_w;
    g.out_h = network_h;
    switch (desc.resize) {
        case GpuPreprocessDescriptor::Resize::LETTERBOX:
        case GpuPreprocessDescriptor::Resize::KEEP_RATIO_PAD_CENTER: {
            const double ratio = std::min(static_cast<double>(network_h) / src_h,
                                           static_cast<double>(network_w) / src_w);
            g.unpad_w = static_cast<int>(std::round(src_w * ratio));
            g.unpad_h = static_cast<int>(std::round(src_h * ratio));
            if (g.unpad_w > network_w) g.unpad_w = network_w;
            if (g.unpad_h > network_h) g.unpad_h = network_h;
            const double dw = (static_cast<double>(network_w) - g.unpad_w) / 2.0;
            const double dh = (static_cast<double>(network_h) - g.unpad_h) / 2.0;
            g.pad_x = std::max(0, static_cast<int>(std::round(dw - 0.1)));
            g.pad_y = std::max(0, static_cast<int>(std::round(dh - 0.1)));
            break;
        }
        case GpuPreprocessDescriptor::Resize::KEEP_RATIO_PAD_ZERO: {
            const double ratio = std::min(static_cast<double>(network_h) / src_h,
                                           static_cast<double>(network_w) / src_w);
            g.unpad_w = static_cast<int>(std::round(src_w * ratio));
            g.unpad_h = static_cast<int>(std::round(src_h * ratio));
            if (g.unpad_w > network_w) g.unpad_w = network_w;
            if (g.unpad_h > network_h) g.unpad_h = network_h;
            break;
        }
        case GpuPreprocessDescriptor::Resize::CENTER_CROP: {
            const int pre_w = desc.pre_crop_size.width > 0 ? desc.pre_crop_size.width : network_w;
            const int pre_h = desc.pre_crop_size.height > 0 ? desc.pre_crop_size.height : network_h;
            g.crop_x = std::max(0, (pre_w - network_w) / 2);
            g.crop_y = std::max(0, (pre_h - network_h) / 2);
            g.unpad_w = pre_w;
            g.unpad_h = pre_h;
            break;
        }
        case GpuPreprocessDescriptor::Resize::DIRECT_RESIZE:
            g.unpad_w = network_w;
            g.unpad_h = network_h;
            break;
        case GpuPreprocessDescriptor::Resize::ALIGN_TO_MULTIPLE:
            g.out_w = ((src_w + desc.align_multiple - 1) / desc.align_multiple) * desc.align_multiple;
            g.out_h = ((src_h + desc.align_multiple - 1) / desc.align_multiple) * desc.align_multiple;
            g.unpad_w = g.out_w;
            g.unpad_h = g.out_h;
            break;
        case GpuPreprocessDescriptor::Resize::NONE:
            g.out_w = src_w;
            g.out_h = src_h;
            g.unpad_w = src_w;
            g.unpad_h = src_h;
            break;
        case GpuPreprocessDescriptor::Resize::DIRECT_RESIZE_PAD_TO_MULTIPLE: {
            const int multiple = desc.align_multiple;
            const int content_w = desc.pre_crop_size.width > 0 ? desc.pre_crop_size.width : src_w;
            const int content_h = desc.pre_crop_size.height > 0 ? desc.pre_crop_size.height : src_h;
            g.unpad_w = content_w;
            g.unpad_h = content_h;
            g.out_w = ((content_w + multiple - 1) / multiple) * multiple;
            g.out_h = ((content_h + multiple - 1) / multiple) * multiple;
            break;
        }
    }
    return g;
}

#endif  // MORTRED_HAS_JPEGGPU

}  // namespace

struct GpuDecodeSlot::State {
#ifdef MORTRED_HAS_JPEGGPU
    jpeggpu_decoder_t decoder = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    void* d_tmp = nullptr;
    size_t d_tmp_cap = 0;
    uint8_t* d_planes[JPEGGPU_MAX_COMP] = {};
    size_t d_plane_caps[JPEGGPU_MAX_COMP] = {};
    uint8_t* h_jpeg = nullptr;
    size_t h_jpeg_cap = 0;
    void* d_output = nullptr;
    size_t d_output_cap = 0;
    void* d_gray = nullptr;
    size_t d_gray_cap = 0;
    bool ok = false;
    bool counted = false;

    ~State() { release(); }

    void release() {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        if (decoder != nullptr) {
            jpeggpu_decoder_cleanup(decoder);
            decoder = nullptr;
        }
        if (d_tmp != nullptr) {
            cudaFree(d_tmp);
            d_tmp = nullptr;
            d_tmp_cap = 0;
        }
        for (int c = 0; c < JPEGGPU_MAX_COMP; ++c) {
            if (d_planes[c] != nullptr) {
                cudaFree(d_planes[c]);
                d_planes[c] = nullptr;
                d_plane_caps[c] = 0;
            }
        }
        if (h_jpeg != nullptr) {
            cudaFreeHost(h_jpeg);
            h_jpeg = nullptr;
            h_jpeg_cap = 0;
        }
        if (d_output != nullptr) {
            cudaFree(d_output);
            d_output = nullptr;
            d_output_cap = 0;
        }
        if (d_gray != nullptr) {
            cudaFree(d_gray);
            d_gray = nullptr;
            d_gray_cap = 0;
        }
        if (event != nullptr) {
            cudaEventDestroy(event);
            event = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (counted) {
            g_open_slots.fetch_sub(1, std::memory_order_relaxed);
            counted = false;
        }
        ok = false;
    }

    bool open() {
        if (ok) {
            return true;
        }
        release();
        if (!probe_jpeggpu()) {
            LOG(INFO) << "gpu jpeg probe: jpeggpu not capable, gpu decode path disabled";
            return false;
        }
        if (jpeggpu_decoder_startup(&decoder) != JPEGGPU_SUCCESS ||
            cudaStreamCreate(&stream) != cudaSuccess ||
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            release();
            LOG(INFO) << "jpeggpu decoder open failed";
            return false;
        }
        ok = true;
        counted = true;
        g_open_slots.fetch_add(1, std::memory_order_relaxed);
        LOG(INFO) << "jpeggpu decoder ready";
        return true;
    }

    GpuPipelineResult decode(const unsigned char* data, size_t size,
                             int network_w, int network_h,
                             const GpuPreprocessDescriptor& desc,
                             std::string* err) {
        GpuPipelineResult out;
        if (err != nullptr) {
            err->clear();
        }
        if (!ok || data == nullptr || size == 0 || !desc.valid()) {
            if (err != nullptr) {
                *err = "jpeggpu slot is not ready";
            }
            return out;
        }
        bool submitted = false;
        auto fail = [&](const char* msg) {
            if (submitted) {
                cudaStreamSynchronize(stream);
            }
            if (err != nullptr) {
                *err = msg;
            }
            return GpuPipelineResult{};
        };

        if (!grow_pinned(&h_jpeg, &h_jpeg_cap, size)) {
            return fail("pinned alloc failed");
        }
        std::memcpy(h_jpeg, data, size);
        struct jpeggpu_img_info info;
        if (jpeggpu_decoder_parse_header(decoder, &info, h_jpeg, size) != JPEGGPU_SUCCESS) {
            return fail("parse_header failed");
        }
        if (info.num_components != 3) {
            return fail("3-component only");
        }
        size_t tmp_size = 0;
        if (jpeggpu_decoder_get_buffer_size(decoder, &tmp_size) != JPEGGPU_SUCCESS) {
            return fail("get_buffer_size failed");
        }
        if (!grow_device(&d_tmp, &d_tmp_cap, tmp_size)) {
            return fail("temp alloc failed");
        }
        if (jpeggpu_decoder_transfer(decoder, d_tmp, tmp_size, stream) != JPEGGPU_SUCCESS) {
            return fail("transfer failed");
        }
        submitted = true;

        struct jpeggpu_img img = {};
        for (int c = 0; c < info.num_components; ++c) {
            img.pitch[c] = ((info.sizes_x[c] + 7) / 8) * 8;
            const size_t plane_bytes = static_cast<size_t>(info.sizes_y[c]) * img.pitch[c];
            void* plane = d_planes[c];
            if (!grow_device(&plane, &d_plane_caps[c], plane_bytes)) {
                return fail("plane alloc failed");
            }
            d_planes[c] = static_cast<uint8_t*>(plane);
            img.image[c] = d_planes[c];
        }
        if (jpeggpu_decoder_decode(decoder, &img, d_tmp, tmp_size, stream) != JPEGGPU_SUCCESS) {
            return fail("decode failed");
        }

        const int src_w = info.sizes_x[0];
        const int src_h = info.sizes_y[0];
        const bool network_blind =
            desc.resize == GpuPreprocessDescriptor::Resize::ALIGN_TO_MULTIPLE ||
            desc.resize == GpuPreprocessDescriptor::Resize::NONE ||
            desc.resize == GpuPreprocessDescriptor::Resize::DIRECT_RESIZE_PAD_TO_MULTIPLE;
        if (src_w <= 0 || src_h <= 0) {
            return fail("invalid image or network size");
        }
        if (!network_blind && (network_w <= 0 || network_h <= 0)) {
            return fail("invalid image or network size");
        }
        if ((desc.resize == GpuPreprocessDescriptor::Resize::ALIGN_TO_MULTIPLE ||
             desc.resize == GpuPreprocessDescriptor::Resize::DIRECT_RESIZE_PAD_TO_MULTIPLE) &&
            desc.align_multiple <= 0) {
            return fail("invalid align_multiple");
        }
        const Geometry geom = compute_geometry(src_w, src_h, network_w, network_h, desc);
        const int channels = desc.color == GpuPreprocessDescriptor::Color::GRAY ? 1 : 3;
        const size_t elem_size = desc.output_dtype == DType::F16 ? 2 : 4;
        const size_t out_bytes = static_cast<size_t>(geom.out_h) * geom.out_w * channels * elem_size;
        if (!grow_device(&d_output, &d_output_cap, out_bytes)) {
            return fail("output buffer alloc failed");
        }
        void* gray = nullptr;
        if (desc.secondary_gray_output) {
            const size_t gray_bytes = static_cast<size_t>(geom.out_h) * geom.out_w * elem_size;
            if (!grow_device(&d_gray, &d_gray_cap, gray_bytes)) {
                return fail("gray output alloc failed");
            }
            gray = d_gray;
        }
        const cudaError_t launch_err = launch_preprocess(
            d_planes[0], d_planes[1], d_planes[2],
            src_w, src_h, info.sizes_x[1], info.sizes_y[1],
            static_cast<int>(img.pitch[0]), static_cast<int>(img.pitch[1]),
            d_output, gray,
            geom.out_w, geom.out_h,
            static_cast<int>(desc.resize), static_cast<int>(desc.color), static_cast<int>(desc.rotation),
            desc.norm.scale,
            desc.norm.mean[0], desc.norm.mean[1], desc.norm.mean[2],
            desc.norm.std[0], desc.norm.std[1], desc.norm.std[2],
            static_cast<float>(desc.pad_value),
            desc.pad_with_mean ? 1 : 0,
            desc.output_dtype == DType::F16 ? 1 : 0,
            desc.output_nhwc ? 1 : 0,
            geom.unpad_w, geom.unpad_h, geom.pad_x, geom.pad_y,
            geom.crop_x, geom.crop_y,
            stream);
        if (launch_err != cudaSuccess) {
            return fail("CUDA kernel launch failed");
        }
        if (cudaEventRecord(event, stream) != cudaSuccess) {
            return fail("frame ready event record failed");
        }

        out.device_input = d_output;
        out.device_gray = gray;
        out.out_w = geom.out_w;
        out.out_h = geom.out_h;
        out.src_w = src_w;
        out.src_h = src_h;
        out.ready_event = event;
        out.valid = true;
        g_request_count[JPEGGPU].fetch_add(1, std::memory_order_relaxed);
        return out;
    }
#else
    bool ok = false;
#endif
};

GpuDecodeSlot::GpuDecodeSlot() = default;
GpuDecodeSlot::~GpuDecodeSlot() = default;

bool GpuDecodeSlot::open() {
#ifndef MORTRED_HAS_JPEGGPU
    return false;
#else
    if (state_ == nullptr) {
        state_ = std::make_unique<State>();
    }
    return state_->open();
#endif
}

void GpuDecodeSlot::close() {
    state_.reset();
}

bool GpuDecodeSlot::ready() const {
    return state_ != nullptr && state_->ok;
}

GpuPipelineResult GpuDecodeSlot::decode_and_preprocess(
    const unsigned char* data, size_t size,
    int network_w, int network_h,
    const GpuPreprocessDescriptor& desc,
    std::string* err) {
#ifndef MORTRED_HAS_JPEGGPU
    (void)data;
    (void)size;
    (void)network_w;
    (void)network_h;
    (void)desc;
    if (err != nullptr) {
        *err = "gpu jpeg decoder not built in the cpu profile";
    }
    return {};
#else
    if (state_ == nullptr || !state_->ok) {
        if (err != nullptr) {
            *err = "jpeggpu slot is not ready";
        }
        return {};
    }
    return state_->decode(data, size, network_w, network_h, desc, err);
#endif
}

std::string render_decode_metrics() {
    static const char* const k_names[BACKEND_COUNT] = {
        "jpeggpu", "cpu-reduced", "cpu-full", "fallback"};
    std::ostringstream ss;
    ss << "# HELP mortred_jpeg_decode_total Images decoded per backend\n";
    ss << "# TYPE mortred_jpeg_decode_total counter\n";
    for (int b = 0; b < BACKEND_COUNT; ++b) {
        const uint64_t n = g_request_count[b].load(std::memory_order_relaxed);
        if (n == 0) {
            continue;
        }
        ss << "mortred_jpeg_decode_total{backend=\"" << k_names[b] << "\"} " << n << "\n";
    }
    ss << "# HELP mortred_jpeg_decode_ladder Workers with a private jpeggpu decoder\n";
    ss << "# TYPE mortred_jpeg_decode_ladder gauge\n";
#ifdef MORTRED_HAS_JPEGGPU
    const bool armed = g_open_slots.load(std::memory_order_relaxed) > 0;
    ss << "mortred_jpeg_decode_ladder{backend=\"" << (armed ? "jpeggpu" : "unavailable")
       << "\",capable=\"" << (armed ? "yes" : "no") << "\"} "
       << g_open_slots.load(std::memory_order_relaxed) << "\n";
#else
    ss << "mortred_jpeg_decode_ladder{backend=\"not-built\",capable=\"no\"} 0\n";
#endif
    return ss.str();
}

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq
