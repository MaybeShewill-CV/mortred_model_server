/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: gpu_jpeg_decoder_stub.cpp
 * Date: 26-9-20
 ************************************************/

// cpu-profile stub: no CUDA / nvjpeg in the dependency tree, so the gpu
// decoder reports unavailable and every caller falls back to cv::imdecode.

#include "models/backend/gpu_jpeg_decoder.h"

namespace jinq {
namespace models {
namespace backend {
namespace gpu_jpeg {

bool recommended() {
    return false;
}

bool available() {
    return false;
}

const char* backend_name() {
    return "not-built";
}

cv::Mat decode(const unsigned char* data, size_t size, std::string* err) {
    (void)data;
    (void)size;
    if (err != nullptr) {
        *err = "gpu jpeg decoder not built in the cpu profile";
    }
    return {};
}

}  // namespace gpu_jpeg
}  // namespace backend
}  // namespace models
}  // namespace jinq
