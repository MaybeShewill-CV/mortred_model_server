/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: trt_spawn_gate.h
* Date: 26-9-5
************************************************/

// Fail-closed spawn check: if the effective model toml uses TensorRT, every
// engine path must exist and be non-empty. Does not deserialize; /ready is
// the loadability probe after prepare.

#ifndef MORTRED_CONTROL_TRT_SPAWN_GATE_H
#define MORTRED_CONTROL_TRT_SPAWN_GATE_H

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#include "control/mini_toml.h"

namespace mortred {
namespace control {

inline constexpr const char* kTrtPrepareHint =
    "run mortredctl prepare (scripts/prepare_pack.sh) on this GPU";

inline std::string trt_gate_error(const std::string& detail) {
    return "TensorRT engine missing or empty: " + detail + "; " + kTrtPrepareHint;
}

inline bool is_trt_gate_error(const std::string& err) {
    return err.find("TensorRT engine missing") != std::string::npos;
}

namespace detail {

inline std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline bool looks_like_backend_section(const std::string& section) {
    return lower_copy(section).find("backend") != std::string::npos;
}

inline bool is_tensorrt_type(const std::string& type) {
    return lower_copy(type) == "tensorrt";
}

inline std::filesystem::path resolve_runtime_file(const std::string& project_root,
                                                  const std::string& bin_dir,
                                                  const std::string& raw) {
    const std::filesystem::path p(raw);
    if (p.is_absolute()) {
        return p.lexically_normal();
    }
    const auto from_bin = (std::filesystem::path(project_root) / bin_dir / p).lexically_normal();
    std::error_code ec;
    if (std::filesystem::is_regular_file(from_bin, ec)) {
        return from_bin;
    }
    const auto from_root = (std::filesystem::path(project_root) / p).lexically_normal();
    if (std::filesystem::is_regular_file(from_root, ec)) {
        return from_root;
    }
    return from_bin;
}

inline std::string server_model_config_path(const std::string& project_root,
                                            const std::string& bin_dir,
                                            const std::string& server_toml) {
    mini_toml::Doc doc;
    if (!mini_toml::load(server_toml, &doc)) {
        return {};
    }
    for (const auto& [section, kv] : doc) {
        if (section.size() > 7 && section.compare(section.size() - 7, 7, "_SERVER") == 0) {
            continue;
        }
        if (kv.count("model_config_file_path") != 0) {
            return resolve_runtime_file(project_root, bin_dir, kv.at("model_config_file_path"))
                .string();
        }
    }
    return {};
}

inline bool tensorrt_cpu_device_error(const mini_toml::Doc& doc, std::string* err) {
    for (const auto& [section, kv] : doc) {
        if (!looks_like_backend_section(section)) {
            continue;
        }
        const std::string type = kv.count("type") != 0 ? kv.at("type") : "";
        if (!is_tensorrt_type(type)) {
            continue;
        }
        const std::string device = kv.count("device") != 0 ? lower_copy(kv.at("device")) : "gpu";
        if (device == "cpu") {
            if (err != nullptr) {
                *err = "tensorrt backend requires device=gpu; device=cpu is a configuration error";
            }
            return true;
        }
    }
    return false;
}

inline std::vector<std::string> trt_engine_raw_paths(const mini_toml::Doc& doc) {
    std::vector<std::string> out;
    for (const auto& [section, kv] : doc) {
        if (!looks_like_backend_section(section)) {
            continue;
        }
        const std::string type = kv.count("type") != 0 ? kv.at("type") : "";
        if (!is_tensorrt_type(type)) {
            continue;
        }
        if (kv.count("model_file_path") == 0 || kv.at("model_file_path").empty()) {
            continue;
        }
        out.push_back(kv.at("model_file_path"));
    }
    return out;
}

}  // namespace detail

/*** Return false and fill err when a TensorRT model would spawn without engines. */
inline bool trt_engines_ready_for_spawn(const std::string& project_root,
                                        const std::string& bin_dir,
                                        const std::string& server_toml,
                                        const std::string& model_config_override,
                                        std::string* err) {
    std::string model_toml = model_config_override;
    if (model_toml.empty()) {
        model_toml = detail::server_model_config_path(project_root, bin_dir, server_toml);
    }
    if (model_toml.empty()) {
        return true;  // catalog already required a model table; nothing to gate
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model_toml, ec)) {
        // Dummy / missing paths are not a TensorRT engine problem. The child
        // still fails at init; crash-loop protection stays on that path.
        return true;
    }
    mini_toml::Doc doc;
    if (!mini_toml::load(model_toml, &doc)) {
        if (err != nullptr) {
            *err = "cannot parse model config: " + model_toml;
        }
        return false;
    }
    if (detail::tensorrt_cpu_device_error(doc, err)) {
        return false;
    }
    const auto raw_paths = detail::trt_engine_raw_paths(doc);
    if (raw_paths.empty()) {
        return true;
    }
    for (const auto& raw : raw_paths) {
        const auto path = detail::resolve_runtime_file(project_root, bin_dir, raw);
        if (!std::filesystem::is_regular_file(path, ec)) {
            if (err != nullptr) {
                *err = trt_gate_error(path.string());
            }
            return false;
        }
        const auto sz = std::filesystem::file_size(path, ec);
        if (ec || sz == 0) {
            if (err != nullptr) {
                *err = trt_gate_error(path.string());
            }
            return false;
        }
    }
    return true;
}

}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_TRT_SPAWN_GATE_H
