/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder.h
 * Date: 26-9-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H
#define MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H

#include <atomic>
#include <cstddef>
#include <string>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {

/*** Backend identifiers for observability (three-layer visibility). */
enum Backend {
    JPEGGPU = 0,
    NVJPEG_HW,
    NVJPEG_SM,
    CPU_REDUCED,
    CPU_FULL,
    FALLBACK,
    BACKEND_COUNT
};

/*** Global request counters per backend, readable by the metrics layer.
 * Incremented by the decode layer on every request. */
extern std::atomic<uint64_t> g_request_count[BACKEND_COUNT];

/*** Name of the currently selected backend (set once at probe time). */
const char* selected_backend_name();

// true when a GPU backend exists AND the multi-image startup race was won
bool recommended();

// true when a GPU backend is merely capable (force mode may still use it)
bool available();

// "jpeggpu" | "nvjpeg-hw" | "nvjpeg-sm" | "unavailable" | "not-built"
const char* backend_name();

/*** full-resolution decode returning a CV_8UC3 BGR Mat. EXIF orientation
 * is NOT applied here (the caller owns it). Fails (empty Mat + err) for
 * anything the GPU path cannot take; callers then retry on the CPU path. */
cv::Mat decode(const unsigned char* data, size_t size, std::string* err);

/*** planar decode result: Y/Cb/Cr as separate CV_8UC1 Mats (jpeggpu native
 * output format, no color conversion applied). Chroma planes may be
 * subsampled (e.g. half size for 4:2:0). Use this + letterbox_ycbcr_nchw
 * to skip the intermediate BGR Mat entirely. */
struct PlanarImage {
    cv::Mat y, cb, cr;  // empty = decode failed
};
PlanarImage decode_planar(const unsigned char* data, size_t size, std::string* err);

/*** device-resident decode result: JPEG decoded to YCbCr planes in GPU
 * memory, NO D2H copy performed. The caller does D2H separately (after
 * the decode timing mark) via fetch_from_device(). */
struct DevicePlanes {
    uint8_t* dev_y = nullptr;
    uint8_t* dev_cb = nullptr;
    uint8_t* dev_cr = nullptr;
    int y_w = 0, y_h = 0;
    int cb_w = 0, cb_h = 0;
    bool valid = false;
};
DevicePlanes decode_to_device(const unsigned char* data, size_t size, std::string* err);
PlanarImage fetch_from_device(const DevicePlanes& dp);

/*** GPU zero-copy pipeline: decode JPEG + letterbox + color convert + fp16 NCHW
 * all on GPU in one pass. Returns a device pointer to the fp16 NCHW tensor that
 * can be passed directly to TRT's input tensor address — no D2H, no CPU
 * preprocess, no H2D. The GPU work is submitted asynchronously; the caller
 * only pays CPU API submission time. */
struct GpuPipelineResult {
    void* device_input = nullptr;  // fp16 NCHW [1,3,H,W] on device
    int out_w = 0, out_h = 0;
    int src_w = 0, src_h = 0;      // original image dims (for postprocess context)
    bool valid = false;
};
GpuPipelineResult decode_and_letterbox_gpu(
    const unsigned char* data, size_t size,
    int network_w, int network_h,
    std::string* err);

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H
