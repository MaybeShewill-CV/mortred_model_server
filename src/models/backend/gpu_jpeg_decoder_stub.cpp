/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder_stub.cpp
 * Date: 26-9-20
 ************************************************/

// cpu-profile stub: no CUDA / nvjpeg / jpeggpu in the dependency tree.

#include "models/backend/gpu_jpeg_decoder.h"

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {

std::atomic<uint64_t> g_request_count[BACKEND_COUNT] = {};

bool recommended() { return false; }
bool available() { return false; }
const char* backend_name() { return "not-built"; }
const char* selected_backend_name() { return "not-built"; }

cv::Mat decode(const unsigned char* data, size_t size, std::string* err) {
    (void)data; (void)size;
    if (err) *err = "gpu jpeg decoder not built in the cpu profile";
    return {};
}

PlanarImage decode_planar(const unsigned char* data, size_t size, std::string* err) {
    (void)data; (void)size;
    if (err) *err = "gpu jpeg decoder not built in the cpu profile";
    return {};
}

GpuPipelineResult decode_and_letterbox_gpu(
    const unsigned char* data, size_t size,
    int network_w, int network_h,
    std::string* err) {
    (void)data; (void)size; (void)network_w; (void)network_h;
    GpuPipelineResult out;
    if (err) *err = "gpu jpeg decoder not built in the cpu profile";
    return out;
}

GpuPipelineResult decode_and_preprocess(
    const unsigned char* data, size_t size,
    int network_w, int network_h,
    const GpuPreprocessDescriptor& desc,
    std::string* err) {
    (void)data; (void)size; (void)network_w; (void)network_h; (void)desc;
    GpuPipelineResult out;
    if (err) *err = "gpu jpeg decoder not built in the cpu profile";
    return out;
}

void release_pipeline_buffers(const GpuPipelineResult& r) {
    (void)r;
}

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq
