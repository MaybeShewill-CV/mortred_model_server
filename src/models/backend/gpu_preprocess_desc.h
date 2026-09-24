/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_preprocess_desc.h
 * Date: 26-9-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_GPU_PREPROCESS_DESC_H
#define MORTRED_MODELS_BACKEND_GPU_PREPROCESS_DESC_H

#include <cstdint>
#include <opencv2/core.hpp>
#include "models/backend/tensor.h"

namespace jinq {
namespace models {
namespace backend {

/*** Declarative description of a model's GPU preprocessing pipeline.
 * Each model registers one via BackendCvModel::set_gpu_preprocess()
 * in its on_init(). The parameterized CUDA kernel reads these fields
 * to produce the model's input tensor directly from decoded YCbCr
 * planes — zero host round-trips.
 *
 * Field order matters for C++ designated initializers (.field = value).
 * The order below matches the usage order in model registrations:
 *   {resize, norm, color, pad_value, output_dtype, output_nhwc, ...}
 */
struct GpuPreprocessDescriptor {
    // ── Geometric transform ──
    enum class Resize {
        LETTERBOX,              // keep-ratio + center pad (YOLO family)
        CENTER_CROP,            // resize to pre_crop_size then crop (classification)
        DIRECT_RESIZE,          // stretch to network size (segmentation/OCR/matting)
        KEEP_RATIO_PAD_ZERO,    // keep-ratio + right/bottom zero pad (DepthAnything)
        KEEP_RATIO_PAD_CENTER,  // keep-ratio + center pad (Metric3D)
        ALIGN_TO_MULTIPLE,      // align up to multiple (CenterFace /32, EnlightenGAN /16)
        NONE,                   // pass through at source resolution (Real-ESRGAN)
        DIRECT_RESIZE_PAD_TO_MULTIPLE // stretch to pre_crop_size (or source), then right/bottom pad to align_multiple (LibFace / YuNet)
    };
    Resize resize = Resize::NONE;

    // ── Normalization: out = (in * scale - mean) / std ──
    struct Norm {
        float scale = 1.0f;
        float mean[3] = {0.0f, 0.0f, 0.0f};
        float std[3] = {1.0f, 1.0f, 1.0f};
    };
    Norm norm;

    // ── Color space ──
    enum class Color { RGB, BGR, GRAY };
    Color color = Color::RGB;

    // ── Padding value (LETTERBOX / KEEP_RATIO) ──
    uint8_t pad_value = 114;
    // ── Pad pixels with the per-channel mean instead of pad_value: the
    // normalized pad reads exactly 0 (Metric3D-style mean padding) ──
    bool pad_with_mean = false;

    // ── Output format ──
    DType output_dtype = DType::F16;  // F16 or F32
    bool output_nhwc = false;          // false = NCHW, true = NHWC

    // ── CENTER_CROP: resize to this size first, then crop to network size.
    // DIRECT_RESIZE_PAD_TO_MULTIPLE: unpadded content size (width, height).
    // Empty means pad the native JPEG size (no stretch). ──
    cv::Size pre_crop_size = {};

    // ── ALIGN_TO_MULTIPLE / DIRECT_RESIZE_PAD_TO_MULTIPLE: the alignment value ──
    int align_multiple = 32;

    // ── Rotation (composable with any Resize) ──
    enum class Rotation { NONE, DEG_90, DEG_180, DEG_270 };
    Rotation rotation = Rotation::NONE;

    // ── Dynamic input size ──
    bool dynamic_size = false;

    // ── Special ──
    bool secondary_gray_output = false;

    bool valid() const {
        return output_dtype == DType::F16 || output_dtype == DType::F32;
    }
};

static_assert(static_cast<int>(GpuPreprocessDescriptor::Resize::DIRECT_RESIZE_PAD_TO_MULTIPLE) == 7,
              "gpu_preprocess.cu RESIZE_* macros must stay in enum order");

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_GPU_PREPROCESS_DESC_H
