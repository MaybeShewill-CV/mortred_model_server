/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder.h
 * Date: 26-9-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H
#define MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H

#include <cstddef>
#include <string>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {

/*** Process-wide nvJPEG decoder (perf iteration 6, S1). The real build lives
 * only in the gpu profile and picks the best backend by probing at first
 * use: NVJPEG_BACKEND_HARDWARE (dedicated NVDEC/JPEG engine), then
 * NVJPEG_BACKEND_GPU (SM decode), then unavailable. cpu-profile builds link
 * a stub that reports unavailable, so callers fall back to cv::imdecode and
 * nothing else changes.
 *
 * All entry points are thread-safe (one decode at a time under a mutex -
 * the decode itself is sub-millisecond GPU work). */

// true when a GPU backend exists AND the startup perf race against
// cv::imdecode on a representative ~1MP jpeg was won by a >=20% margin -
// i.e. "auto" mode should actually use the GPU on this machine
bool recommended();

// true when a GPU backend is merely capable (force mode may still use it)
bool available();

// "hw-nvjpeg" | "sm-nvjpeg" | "sm-nvjpeg(slow,race-lost)" | "unavailable"
// | "not-built"
const char* backend_name();

/*** full-resolution decode returning a CV_8UC3 BGR Mat (EXIF orientation is
 * NOT applied here - the caller owns it, matching cv::imdecode semantics via
 * its own helper). Fails (empty Mat + err) for anything the GPU path cannot
 * take; callers then retry on the CPU path. */
cv::Mat decode(const unsigned char* data, size_t size, std::string* err);

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_GPU_JPEG_DECODER_H
