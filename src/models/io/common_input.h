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

} // namespace common_io
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_COMMON_INPUT_H
