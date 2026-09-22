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
#include <optional>
#include <string>
#include <utility>

#include "toml/toml.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "glog/logging.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/stage_timing.h"
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
    // decode fork policy lives in BackendCvModel (image_decode_backend param:
    // cpu | auto | gpu); this struct only carries W4 reduction facts.
};

/*** Strict (1.0) unless image_decode_mode=budget; optional upscale in (1, 2]. */
inline float parse_image_decode_upscale(const toml::table &params, const char *tag) {
    float decode_upscale = 1.0f;
    const auto decode_mode = params.contains("image_decode_mode") ? params["image_decode_mode"].value<std::string>()
                                                                  : std::optional<std::string>{};
    if (decode_mode.has_value() && *decode_mode == "budget") {
        decode_upscale = 1.25f;
        if (params.contains("image_decode_budget_upscale")) {
            const auto configured = params["image_decode_budget_upscale"].value<double>();
            if (configured.has_value() && *configured > 1.0 && *configured <= 2.0) {
                decode_upscale = static_cast<float>(*configured);
            }
        }
        LOG(INFO) << tag << " reduced JPEG decode enabled (budget_upscale=" << decode_upscale << ")";
    }
    return decode_upscale;
}

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

/*** cv::imdecode flag for the W4 DCT-domain reduced decode */
inline int imread_color_flag_for_reduce(int reduce) {
    switch (reduce) {
        case 2: return cv::IMREAD_REDUCED_COLOR_2;
        case 4: return cv::IMREAD_REDUCED_COLOR_4;
        case 8: return cv::IMREAD_REDUCED_COLOR_8;
        default: return cv::IMREAD_COLOR;
    }
}

/*** minimal JPEG SOF dimension probe (post-rotation via EXIF orientation);
 * returns false for anything that is not a parseable JPEG, and the caller
 * then falls back to a full decode. Optional out params expose the raw EXIF
 * orientation (1-8, 1 = upright; the size is already swap-corrected) and
 * whether the SOF marks a progressive scan (GPU decoder cannot take it).
 * Works on any byte container (vector<unsigned char>, string, ...).
 *
 * Every byte read goes through at(): std::string's element type is SIGNED
 * char, and `bytes[i] != 0xFF` on a signed char is a compile-time constant
 * (char can never hold 255) — without the cast the whole probe folds to
 * `return false` at -O1+ and the GPU zero-copy gate silently dies. */
template <typename Bytes>
inline bool jpeg_full_dimensions_impl(const Bytes &bytes, cv::Size *size, int *orientation_out,
                                 bool *progressive_out) {
    const auto at = [&bytes](size_t i) -> unsigned char {
        return static_cast<unsigned char>(bytes[i]);
    };
    if (bytes.size() < 4 || at(0) != 0xFF || at(1) != 0xD8) {
        return false;
    }
    int width = 0;
    int height = 0;
    int orientation = 1;
    bool progressive = false;
    size_t pos = 2;
    while (pos + 4 <= bytes.size()) {
        if (at(pos) != 0xFF) {
            ++pos;
            continue;
        }
        const unsigned char marker = at(pos + 1);
        if (marker == 0xFF) {
            ++pos;
            continue;
        }
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            pos += 2;
            continue;
        }
        const size_t seg_len = (static_cast<size_t>(at(pos + 2)) << 8) | at(pos + 3);
        if (seg_len < 2) {
            return false;
        }
        if ((marker >= 0xC0 && marker <= 0xC3) || (marker >= 0xC5 && marker <= 0xC7) ||
            (marker >= 0xC9 && marker <= 0xCB) || (marker >= 0xCD && marker <= 0xCF)) {
            if (pos + 9 > bytes.size()) {
                return false;
            }
            height = (at(pos + 5) << 8) | at(pos + 6);
            width = (at(pos + 7) << 8) | at(pos + 8);
            progressive = marker == 0xC2 || marker == 0xC6 || marker == 0xCA || marker == 0xCE;
        } else if (marker == 0xE1 && seg_len >= 16 && pos + 2 + 6 <= bytes.size() &&
                   at(pos + 4) == 'E' && at(pos + 5) == 'x' && at(pos + 6) == 'i' &&
                   at(pos + 7) == 'f') {
            // TIFF header at pos+10; IFD0 entries follow the 8-byte header
            const size_t tiff = pos + 10;
            if (tiff + 8 <= pos + 2 + seg_len && tiff + 8 <= bytes.size()) {
                const bool little = at(tiff) == 0x49 && at(tiff + 1) == 0x49;
                const auto read16 = [&](size_t offset) -> int {
                    const size_t idx = tiff + offset;
                    if (idx + 2 > bytes.size()) {
                        return 0;
                    }
                    return little ? (at(idx) | (at(idx + 1) << 8)) : ((at(idx) << 8) | at(idx + 1));
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

inline bool jpeg_full_dimensions(const std::vector<unsigned char> &bytes, cv::Size *size, int *orientation_out = nullptr,
                                 bool *progressive_out = nullptr) {
    return jpeg_full_dimensions_impl(bytes, size, orientation_out, progressive_out);
}

inline bool jpeg_full_dimensions(const std::string &bytes, cv::Size *size, int *orientation_out = nullptr,
                                 bool *progressive_out = nullptr) {
    return jpeg_full_dimensions_impl(bytes, size, orientation_out, progressive_out);
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

inline int jpeg_reduce_factor(const std::string &bytes, const ImageInputLimits &limits) {
    cv::Size full;
    if (!jpeg_full_dimensions(bytes, &full)) {
        return 1;
    }
    return jpeg_reduce_factor_for(full, limits);
}

/*** planar YCbCr → interleaved BGR (single pass, no intermediate Mats).
 * Used when the GPU decode path returns planar data that needs to be
 * converted to the cv::Mat BGR format the preprocess expects. */

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
/*** image_input -> cv::Mat: the CPU decode path (fork path A). The bytes
 * arrive already transport-decoded (base64 was resolved at bind_parsed_request;
 * raw bodies arrive as-is), so this function is transport-agnostic by
 * construction. May apply the W4 DCT-domain reduction when the limits carry
 * a network hint; full_size (optional) receives the pre-reduction image size
 * so request geometry stays in full-image coordinates. */
inline cv::Mat load_image(const io_define::common_io::image_input &in, const ImageInputLimits &limits, StatusCode *status,
                          std::string *error, cv::Size *full_size = nullptr) {
    if (full_size != nullptr) {
        *full_size = cv::Size();
    }
    const std::string &bytes = in.image.data;
    if (bytes.empty()) {
        if (error != nullptr) {
            *error = "input image data is empty";
        }
        if (status != nullptr) {
            *status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        return {};
    }
    const int reduce = jpeg_reduce_factor(bytes, limits);
    const cv::Mat byte_view(1, static_cast<int>(bytes.size()), CV_8UC1,
                            const_cast<char *>(bytes.data()));
    cv::Mat image = cv::imdecode(byte_view, imread_color_flag_for_reduce(reduce));
    jinq::common::stage_timing::mark("decode");
    jinq::common::stage_timing::annotate("decoder", reduce > 1 ? "libjpeg-turbo-reduced" : "libjpeg-turbo");
    backend::gpu_jpeg::g_request_count[reduce > 1 ? backend::gpu_jpeg::CPU_REDUCED : backend::gpu_jpeg::CPU_FULL]
        .fetch_add(1);
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
