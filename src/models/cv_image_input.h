/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cv_image_input.h
 * Date: 26-8-13
 ************************************************/

#ifndef MORTRED_MODELS_CV_IMAGE_INPUT_H
#define MORTRED_MODELS_CV_IMAGE_INPUT_H

#include <vector>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "glog/logging.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/backend/gpu_jpeg_decoder.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace cv_input {

using jinq::common::StatusCode;

struct ImageInputLimits {
    int64_t max_pixels = 16777216;
    int64_t max_side = 8192;
    // W4 JPEG DCT-domain reduced decode. network_input empty = feature off
    // (all models default). budget_upscale <= 1.0 = strict mode: reduction is
    // only applied when the reduced image still covers the letterbox target
    // (no upsampling). budget_upscale > 1.0 additionally allows one reduction
    // notch whose letterbox upsampling stays within the factor - a numerical
    // accuracy tradeoff that each model opts into via its params.
    cv::Size network_input{};
    float budget_upscale = 1.0f;
    // perf iteration 6 (S1): GPU JPEG decode via nvjpeg. 0=off (default),
    // 1=auto (use GPU only when the startup perf race beat cv::imdecode by
    // >=20%), 2=force (capability-probe only). Per-request fallback to the
    // CPU path covers progressive jpegs, small images and runtime failures.
    int decode_gpu = 0;
    int64_t decode_gpu_min_pixels = 1 << 19;
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

/*** minimal JPEG SOF dimension probe (post-rotation via EXIF orientation);
 * returns false for anything that is not a parseable JPEG, and the caller
 * then falls back to a full decode. Optional out params expose the raw EXIF
 * orientation (1-8, 1 = upright; the size is already swap-corrected) and
 * whether the SOF marks a progressive scan (GPU decoder cannot take it). */
inline bool jpeg_full_dimensions(const std::vector<unsigned char> &bytes, cv::Size *size, int *orientation_out = nullptr,
                                 bool *progressive_out = nullptr) {
    if (bytes.size() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8) {
        return false;
    }
    int width = 0;
    int height = 0;
    int orientation = 1;
    bool progressive = false;
    size_t pos = 2;
    while (pos + 4 <= bytes.size()) {
        if (bytes[pos] != 0xFF) {
            ++pos;
            continue;
        }
        const unsigned char marker = bytes[pos + 1];
        if (marker == 0xFF) {
            ++pos;
            continue;
        }
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            pos += 2;
            continue;
        }
        const size_t seg_len = (static_cast<size_t>(bytes[pos + 2]) << 8) | bytes[pos + 3];
        if (seg_len < 2) {
            return false;
        }
        if ((marker >= 0xC0 && marker <= 0xC3) || (marker >= 0xC5 && marker <= 0xC7) ||
            (marker >= 0xC9 && marker <= 0xCB) || (marker >= 0xCD && marker <= 0xCF)) {
            if (pos + 9 > bytes.size()) {
                return false;
            }
            height = (bytes[pos + 5] << 8) | bytes[pos + 6];
            width = (bytes[pos + 7] << 8) | bytes[pos + 8];
            progressive = marker == 0xC2 || marker == 0xC6 || marker == 0xCA || marker == 0xCE;
        } else if (marker == 0xE1 && seg_len >= 16 && pos + 2 + 6 <= bytes.size() &&
                   bytes[pos + 4] == 'E' && bytes[pos + 5] == 'x' && bytes[pos + 6] == 'i' &&
                   bytes[pos + 7] == 'f') {
            // TIFF header at pos+10; IFD0 entries follow the 8-byte header
            const size_t tiff = pos + 10;
            if (tiff + 8 <= pos + 2 + seg_len && tiff + 8 <= bytes.size()) {
                const bool little = bytes[tiff] == 0x49 && bytes[tiff + 1] == 0x49;
                const auto read16 = [&](size_t offset) -> int {
                    const size_t at = tiff + offset;
                    if (at + 2 > bytes.size()) {
                        return 0;
                    }
                    return little ? (bytes[at] | (bytes[at + 1] << 8)) : ((bytes[at] << 8) | bytes[at + 1]);
                };
                const int entry_count = read16(4);
                for (int entry = 0; entry < entry_count && entry < 64; ++entry) {
                    const size_t base = 8 + static_cast<size_t>(entry) * 12;
                    if (read16(base) == 0x0112) {
                        orientation = read16(base + 8);
                        break;
                    }
                }
            }
        }
        if (width > 0 && height > 0) {
            break;
        }
        pos += 2 + seg_len;
    }
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (orientation >= 5 && orientation <= 8) {
        std::swap(width, height);
    }
    *size = cv::Size(width, height);
    if (orientation_out != nullptr) {
        *orientation_out = orientation;
    }
    if (progressive_out != nullptr) {
        *progressive_out = progressive;
    }
    return true;
}

/*** reduction factor in {1,2,4,8} for the DCT-domain decode, from a known
 * full size. min(N/W, N/H) is invariant under a W/H swap, so EXIF rotation
 * cannot invalidate the strict-mode bound. Returns 1 (= off) whenever the
 * hint is unset. */
inline int jpeg_reduce_factor_for(const cv::Size &full, const ImageInputLimits &limits) {
    if (limits.network_input.width <= 0 || limits.network_input.height <= 0) {
        return 1;
    }
    const double s = std::min(static_cast<double>(limits.network_input.width) / static_cast<double>(full.width),
                              static_cast<double>(limits.network_input.height) / static_cast<double>(full.height));
    int factor = 1;
    for (int candidate : {2, 4, 8}) {
        if (static_cast<double>(candidate) <= 1.0 / s) {
            factor = candidate;
        }
    }
    if (limits.budget_upscale > 1.0f && factor < 8) {
        const int bumped = factor * 2;
        if (static_cast<double>(bumped) * s <= static_cast<double>(limits.budget_upscale)) {
            factor = bumped;
        }
    }
    return factor;
}

inline int jpeg_reduce_factor(const std::vector<unsigned char> &bytes, const ImageInputLimits &limits) {
    cv::Size full;
    if (!jpeg_full_dimensions(bytes, &full)) {
        return 1;
    }
    return jpeg_reduce_factor_for(full, limits);
}

/*** applies the raw EXIF orientation the way cv::imdecode does internally;
 * only the GPU decode path needs it (the swap-corrected size already comes
 * from jpeg_full_dimensions) */
inline void apply_exif_orientation(cv::Mat *image, int orientation) {
    if (image == nullptr || image->empty() || orientation <= 1 || orientation > 8) {
        return;
    }
    switch (orientation) {
        case 2:
            cv::flip(*image, *image, 1);
            break;
        case 3:
            cv::rotate(*image, *image, cv::ROTATE_180);
            break;
        case 4:
            cv::flip(*image, *image, 0);
            break;
        case 5:
            cv::transpose(*image, *image);
            break;
        case 6:
            cv::rotate(*image, *image, cv::ROTATE_90_CLOCKWISE);
            break;
        case 7:
            cv::flip(*image, *image, -1);
            cv::transpose(*image, *image);
            break;
        case 8:
        default:
            cv::rotate(*image, *image, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
    }
}

inline int imread_color_flag_for_reduce(int factor) {
    switch (factor) {
        case 2:
            return cv::IMREAD_REDUCED_COLOR_2;
        case 4:
            return cv::IMREAD_REDUCED_COLOR_4;
        case 8:
            return cv::IMREAD_REDUCED_COLOR_8;
        default:
            return cv::IMREAD_COLOR;
    }
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
 * is decoded first, raw bytes are used as-is; both then share one decode
 * path which may apply the W4 DCT-domain reduction when the limits carry a
 * network hint. full_size (optional) receives the pre-reduction image size
 * so request geometry stays in full-image coordinates.
 */
inline cv::Mat load_image(const io_define::common_io::image_input &in, const ImageInputLimits &limits, StatusCode *status,
                          std::string *error, cv::Size *full_size = nullptr) {
    if (full_size != nullptr) {
        *full_size = cv::Size();
    }
    std::vector<unsigned char> bytes;
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
        bytes.assign(in.image.data.begin(), in.image.data.end());
    } else {
        const std::string decoded = jinq::common::base64::decode(in.image.data);
        if (decoded.empty()) {
            if (error != nullptr) {
                *error = "input image base64 data is empty or invalid";
            }
            if (status != nullptr) {
                *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
            }
            return {};
        }
        bytes.assign(decoded.begin(), decoded.end());
    }
    const int reduce = jpeg_reduce_factor(bytes, limits);
    // perf iteration 6 (S1): GPU decode bypass - full-resolution nvjpeg
    // decode (the fused letterbox kernel then resizes straight from the
    // full frame). auto mode honors the startup perf race; force mode only
    // checks capability. Any miss falls through to the CPU path.
    const bool gpu_allowed = (limits.decode_gpu == 2 && backend::gpu_jpeg::available()) ||
                             (limits.decode_gpu == 1 && backend::gpu_jpeg::recommended());
    if (gpu_allowed) {
        cv::Size full;
        int orientation = 1;
        bool progressive = false;
        if (jpeg_full_dimensions(bytes, &full, &orientation, &progressive) && !progressive &&
            static_cast<int64_t>(full.width) * static_cast<int64_t>(full.height) >= limits.decode_gpu_min_pixels) {
            std::string gpu_error;
            cv::Mat gpu_image = backend::gpu_jpeg::decode(bytes.data(), bytes.size(), &gpu_error);
            if (!gpu_image.empty()) {
                apply_exif_orientation(&gpu_image, orientation);
                if (image_within_limits(gpu_image, limits, error)) {
                    cv::Mat ret = normalize_to_bgr8uc3(gpu_image, error);
                    if (!ret.empty()) {
                        return ret;
                    }
                }
            } else {
                LOG_EVERY_N(WARNING, 100) << "gpu jpeg decode fell back to cpu: " << gpu_error;
            }
        }
    }
    cv::Mat image = cv::imdecode(bytes, imread_color_flag_for_reduce(reduce));
    if (image.empty()) {
        if (error != nullptr) {
            *error = "input image bytes are not a decodable image";
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
    if (reduce > 1 && full_size != nullptr) {
        cv::Size full;
        if (jpeg_full_dimensions(bytes, &full)) {
            *full_size = full;
        }
    }
    cv::Mat ret = normalize_to_bgr8uc3(image, error);
    if (ret.empty() && status != nullptr) {
        *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    return ret;
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
