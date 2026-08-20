/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: backend/backend_config.h
 * Date: 2026-08-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_BACKEND_CONFIG_H
#define MORTRED_MODELS_BACKEND_BACKEND_CONFIG_H

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace backend {

/***
 * Unified backend selection block, parsed from the [MODEL.backend] table of
 * the new model config schema:
 *
 *   [MODEL.backend]
 *   type = "mnn" | "onnx" | "tensorrt"
 *   model_file_path = "..."
 *   device = "cpu" | "cuda"
 *   device_id = 0
 *   threads = 4
 *   precision_mode = 0     # mnn only
 *   power_mode = 0         # mnn only
 *   input_layout = "auto" | "nhwc" | "nchw"   # mnn only
 *   input_names = ["..."]  # optional, defaults to the model file io
 *   output_names = ["..."] # optional, defaults to the model file io
 */
struct BackendConfig {
    std::string type;
    std::string model_file_path;
    std::string device = "cpu";
    int device_id = 0;
    int threads = 4;
    int precision_mode = 0;
    int power_mode = 0;
    std::string input_layout = "auto";
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    bool is_mnn() const { return type == "mnn"; }
    bool is_onnx() const { return type == "onnx"; }
    bool is_tensorrt() const { return type == "tensorrt"; }
    bool use_cuda() const { return device == "cuda"; }
};

namespace detail {

inline bool is_known_backend(const std::string& type) {
    return type == "mnn" || type == "onnx" || type == "tensorrt";
}

template<typename T>
bool read_int(const toml::table& table, const std::string& key, T* out, std::string* err) {
    if (!table.contains(key)) {
        return true;
    }
    const auto value = table[key].value_or<int64_t>(INT64_MIN);
    if (value == INT64_MIN || value < std::numeric_limits<T>::min() ||
        value > std::numeric_limits<T>::max()) {
        if (err != nullptr) {
            *err = "backend key '" + key + "' must be an integer";
        }
        return false;
    }
    *out = static_cast<T>(value);
    return true;
}

inline bool read_string(const toml::table& table, const std::string& key, std::string* out,
                        std::string* err, bool allow_empty = true) {
    if (!table.contains(key)) {
        return true;
    }
    if (!table[key].is_string()) {
        if (err != nullptr) {
            *err = "backend key '" + key + "' must be a string";
        }
        return false;
    }
    *out = table[key].value_or<std::string>("");
    if (!allow_empty && out->empty()) {
        if (err != nullptr) {
            *err = "backend key '" + key + "' must be a non-empty string";
        }
        return false;
    }
    return true;
}

inline bool read_string_array(const toml::table& table, const std::string& key,
                              std::vector<std::string>* out, std::string* err) {
    if (!table.contains(key)) {
        return true;
    }
    const toml::array* array = table[key].as_array();
    if (array == nullptr) {
        if (err != nullptr) {
            *err = "backend key '" + key + "' must be an array of strings";
        }
        return false;
    }
    out->clear();
    out->reserve(array->size());
    for (size_t idx = 0; idx < array->size(); ++idx) {
        const auto value = (*array)[idx].value_or<std::string>("");
        if (value.empty()) {
            if (err != nullptr) {
                *err = "backend key '" + key + "' must contain non-empty strings";
            }
            return false;
        }
        out->push_back(value);
    }
    return true;
}

}  // namespace detail

/***
 * Parse and validate one backend table (either the [MODEL.backend] table or
 * an extra [<key>_backend] table of a multi-session model).
 */
inline bool parse_backend_table(const toml::table& backend_table, BackendConfig* out,
                                std::string* err) {
    if (err != nullptr) {
        err->clear();
    }

    BackendConfig config;
    if (!detail::read_string(backend_table, "type", &config.type, err, false)) {
        return false;
    }
    if (config.type.empty()) {
        if (err != nullptr) {
            *err = "backend key 'type' is required (mnn | onnx | tensorrt)";
        }
        return false;
    }
    if (!detail::is_known_backend(config.type)) {
        if (err != nullptr) {
            *err = "unknown backend type '" + config.type + "', expected mnn | onnx | tensorrt";
        }
        return false;
    }
    if (!detail::read_string(backend_table, "model_file_path", &config.model_file_path, err,
                             false)) {
        return false;
    }
    if (config.model_file_path.empty()) {
        if (err != nullptr) {
            *err = "backend key 'model_file_path' is required";
        }
        return false;
    }
    if (!detail::read_string(backend_table, "device", &config.device, err)) {
        return false;
    }
    if (!config.device.empty() && config.device != "cpu" && config.device != "cuda") {
        if (err != nullptr) {
            *err = "backend key 'device' must be 'cpu' or 'cuda', got '" + config.device + "'";
        }
        return false;
    }
    if (config.device.empty()) {
        config.device = "cpu";
    }
    if (!detail::read_int(backend_table, "device_id", &config.device_id, err) ||
        !detail::read_int(backend_table, "gpu_device_id", &config.device_id, err) ||
        !detail::read_int(backend_table, "threads", &config.threads, err) ||
        !detail::read_int(backend_table, "precision_mode", &config.precision_mode, err) ||
        !detail::read_int(backend_table, "power_mode", &config.power_mode, err)) {
        return false;
    }
    if (config.threads <= 0) {
        if (err != nullptr) {
            *err = "backend key 'threads' must be positive";
        }
        return false;
    }
    if (!detail::read_string(backend_table, "input_layout", &config.input_layout, err)) {
        return false;
    }
    if (config.input_layout.empty()) {
        config.input_layout = "auto";
    }
    if (config.input_layout != "auto" && config.input_layout != "nhwc" &&
        config.input_layout != "nchw") {
        if (err != nullptr) {
            *err = "backend key 'input_layout' must be auto | nhwc | nchw, got '"
                   + config.input_layout + "'";
        }
        return false;
    }
    if (!detail::read_string_array(backend_table, "input_names", &config.input_names, err) ||
        !detail::read_string_array(backend_table, "output_names", &config.output_names, err)) {
        return false;
    }

    if (out != nullptr) {
        *out = std::move(config);
    }
    return true;
}

/***
 * Parse and validate the [MODEL.backend] sub-table of a model section.
 * Returns false and fills err on contract violations.
 */
inline bool parse_backend_config(const toml::table& model_section, BackendConfig* out,
                                 std::string* err) {
    if (err != nullptr) {
        err->clear();
    }
    if (!model_section.contains("backend")) {
        if (err != nullptr) {
            *err = "model section does not contain the [backend] sub-table";
        }
        return false;
    }
    const toml::table* backend_table = model_section["backend"].as_table();
    if (backend_table == nullptr) {
        if (err != nullptr) {
            *err = "[backend] must be a table";
        }
        return false;
    }
    return parse_backend_table(*backend_table, out, err);
}

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_BACKEND_CONFIG_H
