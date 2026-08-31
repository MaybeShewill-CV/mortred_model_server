/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: common_input.h
 * Date: 26-8-30
 ************************************************/

#ifndef MORTRED_MODELS_IO_COMMON_INPUT_H
#define MORTRED_MODELS_IO_COMMON_INPUT_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace jinq {
namespace models {
namespace backend {

// forward declaration: image_input carries a request-scoped parameter view;
// the request owns the ParamSet, the input only borrows it for one run
class ParamSet;

} // namespace backend

namespace io_define {
namespace common_io {

// shared input types: raw mat / file path / base64 / image pair

struct mat_input {
    cv::Mat input_image;
};

struct file_input {
    std::string input_image_path;
};

struct base64_input {
    std::string input_image_content;
};

struct pair_mat_input {
    cv::Mat src_input_image;
    cv::Mat dst_input_image;
};

/*** transport-agnostic image payload: how the bytes arrive. base64_text is
 * the JSON-envelope encoding; raw_bytes is the binary-body encoding (body
 * bytes go straight to imdecode, no base64 inflation). The request owns the
 * buffer, the model only reads it. */
struct byte_source {
    enum class origin_kind { base64_text, raw_bytes };

    origin_kind origin = origin_kind::base64_text;
    std::string data;
};

/*** unified request-scoped image input: one image plus the request-level
 * parameter view. params is nullptr on the legacy single-image path, so a
 * model reading it falls back to its config defaults unchanged. */
struct image_input {
    byte_source image;
    const backend::ParamSet *params = nullptr;
};

} // namespace common_io
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_COMMON_INPUT_H
