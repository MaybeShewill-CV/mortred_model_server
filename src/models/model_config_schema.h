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
        if (key == "model_file_path" || key == "class_name_file") {
            if (!value.is_string() || value.value_or<std::string>("").empty()) {
                return fail("key '" + key + "' must be a non-empty string");
            }
        } else if (key == "model_threads_num" || key == "backend_precision_mode" ||
                   key == "backend_power_mode" || key == "compute_threads") {
            if (!value.is_integer()) {
                return fail("key '" + key + "' must be an integer");
            }
            if (key == "model_threads_num" && value.value_or<int64_t>(0) <= 0) {
                return fail("key 'model_threads_num' must be positive");
            }
        } else if (key == "compute_backend") {
            if (!value.is_string()) {
                return fail("key 'compute_backend' must be a string");
            }
            const std::string backend = value.value_or<std::string>("");
            if (backend != "cpu" && backend != "cuda") {
                return fail("key 'compute_backend' must be 'cpu' or 'cuda', got '" + backend + "'");
            }
        } else if (key == "model_input_image_size") {
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
