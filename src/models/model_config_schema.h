/************************************************
 * Author: Codex
 * File: model_config_schema.h
 *
 * Contract validator for model config sections (the flat [MODEL] blocks
 * consumed by MnnNet). Model configs legitimately carry model-specific keys
 * (diffusion samplers, TRT backends, trackers, ...), so unlike the server
 * schema this validator only enforces the *common MNN block* types and lets
 * every other key pass. A missing model_file_path is already rejected by
 * MnnNet::init at runtime.
 ************************************************/

#ifndef MORTRED_MODEL_CONFIG_SCHEMA_H
#define MORTRED_MODEL_CONFIG_SCHEMA_H

#include <string>
#include <vector>

#include <toml/toml.hpp>

namespace jinq {
namespace models {

/***
 * Validate the common MNN block of a model config section. Returns false and
 * fills err on type/value violations of the shared keys; all other keys are
 * model-specific and out of the common contract.
 */
inline bool validate_model_config_section(const toml::table& section,
                                          std::string* err,
                                          std::vector<std::string>* warnings = nullptr) {
    auto fail = [err](const std::string& message) {
        if (err != nullptr) {
            *err = message;
        }
        return false;
    };
    (void)warnings;

    for (const auto& [key, value] : section) {
        // toml::v3::key 无到 std::string 的隐式转换（str() 返回 string_view），
        // 错误消息统一使用物化的 key_name
        const std::string key_name(key.str());
        // 注意：此 toml11 版本中 key == "literal" 恒为 false，必须用 key_name 比较
        if (key_name == "model_file_path" || key_name == "class_name_file") {
            if (!value.is_string() || value.value_or<std::string>("").empty()) {
                return fail("key '" + key_name + "' must be a non-empty string");
            }
        } else if (key_name == "model_threads_num" || key_name == "backend_precision_mode" ||
                   key_name == "backend_power_mode" || key_name == "compute_threads") {
            if (!value.is_integer()) {
                return fail("key '" + key_name + "' must be an integer");
            }
            if (key_name == "model_threads_num" && value.value_or<int64_t>(0) <= 0) {
                return fail("key 'model_threads_num' must be positive");
            }
        } else if (key_name == "compute_backend") {
            if (!value.is_string()) {
                return fail("key 'compute_backend' must be a string");
            }
            const std::string backend = value.value_or<std::string>("");
            if (backend != "cpu" && backend != "cuda") {
                return fail("key 'compute_backend' must be 'cpu' or 'cuda', got '" + backend + "'");
            }
        } else if (key_name == "model_input_image_size") {
            if (!value.is_array()) {
                return fail("key 'model_input_image_size' must be an array of two integers");
            }
        }
        // all other keys are model-specific and out of the common contract
    }
    return true;
}

}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODEL_CONFIG_SCHEMA_H
