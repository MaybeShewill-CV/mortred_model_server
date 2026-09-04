#ifndef MORTRED_MODELS_OBJECT_DETECTION_DETECTION_PARAMS_H
#define MORTRED_MODELS_OBJECT_DETECTION_DETECTION_PARAMS_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace object_detection {

struct DetectionParams {
    float score_threshold = 0.4f;
    float nms_threshold = 0.35f;
    int keep_top_k = 250;
    int class_nums = 80;
    float min_box_area_px = 5.0f;
    bool clip_boxes = false;
    std::vector<std::string> class_names;

    static bool parse(const toml::table &params, DetectionParams *out, std::string *error);
};

inline bool DetectionParams::parse(const toml::table &params, DetectionParams *out, std::string *error) {
    if (out == nullptr) {
        if (error != nullptr) {
            *error = "DetectionParams::parse: output pointer is null";
        }
        return false;
    }
    if (params.contains("input_node_size")) {
        if (error != nullptr) {
            *error = "params key 'input_node_size' is not supported; use "
                     "'model_input_image_size = [height, width]'";
        }
        return false;
    }
    if (params.contains("model_score_threshold")) {
        const auto value = params["model_score_threshold"].value_or<double>(-1.0);
        if (value < 0.0 || value > 1.0) {
            if (error != nullptr) {
                *error = "params key 'model_score_threshold' must be in [0, 1], got " + std::to_string(value);
            }
            return false;
        }
        out->score_threshold = static_cast<float>(value);
    }
    if (params.contains("model_nms_threshold")) {
        const auto value = params["model_nms_threshold"].value_or<double>(-1.0);
        if (value < 0.0 || value > 1.0) {
            if (error != nullptr) {
                *error = "params key 'model_nms_threshold' must be in [0, 1], got " + std::to_string(value);
            }
            return false;
        }
        out->nms_threshold = static_cast<float>(value);
    }
    if (params.contains("model_keep_top_k")) {
        const auto value = params["model_keep_top_k"].value_or<int64_t>(-1);
        if (value < 1 || value > 10000) {
            if (error != nullptr) {
                *error = "params key 'model_keep_top_k' must be in [1, 10000], got " + std::to_string(value);
            }
            return false;
        }
        out->keep_top_k = static_cast<int>(value);
    }
    if (params.contains("model_class_nums")) {
        const auto value = params["model_class_nums"].value_or<int64_t>(-1);
        if (value < 1) {
            if (error != nullptr) {
                *error = "params key 'model_class_nums' must be >= 1, got " + std::to_string(value);
            }
            return false;
        }
        out->class_nums = static_cast<int>(value);
    }
    if (params.contains("min_box_area_px")) {
        const auto value = params["min_box_area_px"].value_or<double>(-1.0);
        if (value < 0.0) {
            if (error != nullptr) {
                *error = "params key 'min_box_area_px' must be >= 0, got " + std::to_string(value);
            }
            return false;
        }
        out->min_box_area_px = static_cast<float>(value);
    }
    if (params.contains("clip_boxes")) {
        if (!params["clip_boxes"].is_boolean()) {
            if (error != nullptr) {
                *error = "params key 'clip_boxes' must be a boolean";
            }
            return false;
        }
        const bool value = params["clip_boxes"].value_or<bool>(false);
        out->clip_boxes = value;
    }
    if (params.contains("class_names")) {
        const toml::array *names = params["class_names"].as_array();
        if (names == nullptr) {
            if (error != nullptr) {
                *error = "params key 'class_names' must be an array of non-empty strings";
            }
            return false;
        }
        out->class_names.clear();
        out->class_names.reserve(names->size());
        for (size_t idx = 0; idx < names->size(); ++idx) {
            const auto value = (*names)[idx].value_or<std::string>("");
            if (value.empty()) {
                if (error != nullptr) {
                    *error = "params key 'class_names' contains a non-string or empty entry at " + std::to_string(idx);
                }
                return false;
            }
            out->class_names.push_back(value);
        }
        if (out->class_names.size() != static_cast<size_t>(out->class_nums)) {
            if (error != nullptr) {
                *error = "params key 'class_names' has " + std::to_string(out->class_names.size()) + " entries, expected " +
                         std::to_string(out->class_nums);
            }
            return false;
        }
    }
    return true;
}

inline bool parse_model_input_size(const toml::table &params, cv::Size *out, std::string *error) {
    if (!params.contains("model_input_image_size")) {
        return true;
    }
    const toml::array *size = params["model_input_image_size"].as_array();
    if (size == nullptr || size->size() != 2) {
        if (error != nullptr) {
            *error = "params key 'model_input_image_size' must be [height, width]";
        }
        return false;
    }
    const int64_t height = (*size)[0].value_or<int64_t>(-1);
    const int64_t width = (*size)[1].value_or<int64_t>(-1);
    if (height <= 0 || width <= 0) {
        if (error != nullptr) {
            *error = "params key 'model_input_image_size' must contain positive integers";
        }
        return false;
    }
    *out = cv::Size(static_cast<int>(width), static_cast<int>(height));
    return true;
}

} // namespace object_detection
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_OBJECT_DETECTION_DETECTION_PARAMS_H
