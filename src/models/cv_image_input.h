/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cv_image_input.h
 * Date: 26-8-13
 ************************************************/

#ifndef MORTRED_MODELS_CV_IMAGE_INPUT_H
#define MORTRED_MODELS_CV_IMAGE_INPUT_H

#include <vector>

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace cv_input {

using jinq::common::StatusCode;

struct ImageInputLimits {
    int64_t max_pixels = 16777216;
    int64_t max_side = 8192;
};

inline bool image_within_limits(const cv::Mat &image, const ImageInputLimits &limits, std::string *error) {
    if (image.empty()) {
        if (error != nullptr) {
            *error = "input image is empty";
        }
        return false;
    }
    if (image.rows <= 0 || image.cols <= 0) {
        if (error != nullptr) {
            *error = "input image has invalid dimensions";
        }
        return false;
    }
    if (image.rows > limits.max_side || image.cols > limits.max_side) {
        if (error != nullptr) {
            *error = "input image side exceeds limit: " + std::to_string(image.cols) + "x" + std::to_string(image.rows) +
                     ", max_side=" + std::to_string(limits.max_side);
        }
        return false;
    }
    const int64_t pixels = static_cast<int64_t>(image.rows) * image.cols;
    if (pixels > limits.max_pixels) {
        if (error != nullptr) {
            *error = "input image has " + std::to_string(pixels) + " pixels, max_pixels=" + std::to_string(limits.max_pixels);
        }
        return false;
    }
    return true;
}

inline cv::Mat normalize_to_bgr8uc3(const cv::Mat &image, std::string *error) {
    if (image.empty()) {
        if (error != nullptr) {
            *error = "input image is empty";
        }
        return {};
    }
    cv::Mat bgr;
    if (image.type() == CV_8UC3) {
        bgr = image;
    } else if (image.type() == CV_8UC1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.type() == CV_8UC4) {
        // OpenCV image decoding produces BGRA for four-channel images. Other
        // producers must convert to one of the three explicitly supported Mats.
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else {
        if (error != nullptr) {
            *error = "unsupported input Mat type " + std::to_string(image.type()) + ", expected CV_8UC1/CV_8UC3/CV_8UC4";
        }
        return {};
    }
    return bgr;
}

inline StatusCode status_for_image_load(const std::string &error) {
    if (error.find("exceeds limit") != std::string::npos || error.find("pixels, max_pixels=") != std::string::npos) {
        return StatusCode::REQUEST_ENTITY_TOO_LARGE;
    }
    return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
}

/***
 * file_input -> cv::Mat: reads with original channels after existence check
 */
inline cv::Mat load_image(const io_define::common_io::file_input &in, const ImageInputLimits &limits, StatusCode *status,
                          std::string *error) {
    cv::Mat ret;
    if (!jinq::common::FilePathUtil::is_file_exist(in.input_image_path)) {
        if (error != nullptr) {
            *error = "input image: " + in.input_image_path + " not exist";
        }
        if (status != nullptr) {
            *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        return ret;
    }
    cv::Mat decoded = cv::imread(in.input_image_path, cv::IMREAD_COLOR);
    if (!image_within_limits(decoded, limits, error)) {
        if (status != nullptr) {
            *status = status_for_image_load(error == nullptr ? "" : *error);
        }
        return {};
    }
    ret = normalize_to_bgr8uc3(decoded, error);
    if (ret.empty() && status != nullptr) {
        *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    return ret;
}

/***
 * mat_input -> cv::Mat: refcounted shallow copy, zero overhead
 */
inline cv::Mat load_image(const io_define::common_io::mat_input &in, const ImageInputLimits &limits, StatusCode *status,
                          std::string *error) {
    if (!image_within_limits(in.input_image, limits, error)) {
        if (status != nullptr) {
            *status = status_for_image_load(error == nullptr ? "" : *error);
        }
        return {};
    }
    cv::Mat ret = normalize_to_bgr8uc3(in.input_image, error);
    if (ret.empty() && status != nullptr) {
        *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    return ret;
}

/***
 * base64_input -> cv::Mat: base64 decode then decode as three-channel BGR
 */
inline cv::Mat load_image(const io_define::common_io::base64_input &in, const ImageInputLimits &limits, StatusCode *status,
                          std::string *error) {
    const std::string decoded = jinq::common::base64::decode(in.input_image_content);
    if (decoded.empty()) {
        if (error != nullptr) {
            *error = "input image base64 data is empty or invalid";
        }
        if (status != nullptr) {
            *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        return {};
    }
    std::vector<unsigned char> bytes(decoded.begin(), decoded.end());
    cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (!image_within_limits(image, limits, error)) {
        if (status != nullptr) {
            *status = status_for_image_load(error == nullptr ? "" : *error);
        }
        return {};
    }
    cv::Mat ret = normalize_to_bgr8uc3(image, error);
    if (ret.empty() && status != nullptr) {
        *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    return ret;
}

/***
 * image_input -> cv::Mat: dispatches on the byte_source origin. base64 text
 * reuses the base64_input path unchanged; raw bytes are fed straight to
 * imdecode with no base64 inflation (binary body encoding).
 */
inline cv::Mat load_image(const io_define::common_io::image_input &in, const ImageInputLimits &limits, StatusCode *status,
                          std::string *error) {
    if (in.image.origin == io_define::common_io::byte_source::origin_kind::raw_bytes) {
        if (in.image.data.empty()) {
            if (error != nullptr) {
                *error = "input image raw data is empty";
            }
            if (status != nullptr) {
                *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
            }
            return {};
        }
        std::vector<unsigned char> bytes(in.image.data.begin(), in.image.data.end());
        cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
        if (image.empty()) {
            if (error != nullptr) {
                *error = "input image raw bytes are not a decodable image";
            }
            if (status != nullptr) {
                *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
            }
            return {};
        }
        if (!image_within_limits(image, limits, error)) {
            if (status != nullptr) {
                *status = status_for_image_load(error == nullptr ? "" : *error);
            }
            return {};
        }
        cv::Mat ret = normalize_to_bgr8uc3(image, error);
        if (ret.empty() && status != nullptr) {
            *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        return ret;
    }
    io_define::common_io::base64_input base64;
    base64.input_image_content = in.image.data;
    return load_image(base64, limits, status, error);
}

inline cv::Mat load_image(const io_define::common_io::file_input &in) {
    return cv_input::load_image(in, ImageInputLimits{}, nullptr, nullptr);
}

inline cv::Mat load_image(const io_define::common_io::mat_input &in) {
    return cv_input::load_image(in, ImageInputLimits{}, nullptr, nullptr);
}

inline cv::Mat load_image(const io_define::common_io::base64_input &in) {
    return cv_input::load_image(in, ImageInputLimits{}, nullptr, nullptr);
}

inline cv::Mat load_image(const io_define::common_io::image_input &in) {
    return cv_input::load_image(in, ImageInputLimits{}, nullptr, nullptr);
}

} // namespace cv_input
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_CV_IMAGE_INPUT_H
