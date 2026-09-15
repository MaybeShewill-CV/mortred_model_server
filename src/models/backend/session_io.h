/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: session_io.h
 * Date: 26-9-8
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_SESSION_IO_H
#define MORTRED_MODELS_BACKEND_SESSION_IO_H

#include <algorithm>
#include <string>
#include <vector>

#include "common/status_code.h"
#include "models/backend/session.h"

namespace jinq {
namespace models {
namespace backend {

/***
 * Apply optional [MODEL.backend] input_names / output_names to a backend's
 * discovered TensorInfo list.
 *
 * Empty names: keep the backend discovery order (MNN map order, ONNX graph
 * order, TensorRT engine IO order). That order is not stable across backends,
 * so multi-output models must set output_names or read tensors by name.
 *
 * Non-empty names: keep only those tensors, in the configured vector order.
 * Missing or duplicated names fail init. All three backends share this rule.
 */
inline StatusCode apply_configured_io_names(const std::vector<std::string>& names,
                                            std::vector<TensorInfo>* infos, const char* io_kind,
                                            std::string* err) {
    if (infos == nullptr) {
        if (err != nullptr) {
            *err = "internal error: io info list is null";
        }
        return StatusCode::MODEL_INIT_FAILED;
    }
    const char* kind = (io_kind == nullptr || io_kind[0] == '\0') ? "io" : io_kind;
    if (names.empty()) {
        return StatusCode::OK;
    }

    std::vector<TensorInfo> selected;
    selected.reserve(names.size());
    for (const auto& name : names) {
        if (name.empty()) {
            if (err != nullptr) {
                *err = std::string("configured ") + kind + " name is empty";
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
        const auto already = std::find_if(
            selected.begin(), selected.end(),
            [&name](const TensorInfo& info) { return info.name == name; });
        if (already != selected.end()) {
            if (err != nullptr) {
                *err = std::string("configured ") + kind + " name duplicated: " + name;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
        const auto found = std::find_if(
            infos->begin(), infos->end(),
            [&name](const TensorInfo& info) { return info.name == name; });
        if (found == infos->end()) {
            if (err != nullptr) {
                *err = std::string("configured ") + kind + " tensor not found: " + name;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
        selected.push_back(*found);
    }
    *infos = std::move(selected);
    return StatusCode::OK;
}


/***
 * Resolve run() inputs against the session's required TensorInfo list.
 * Fail-closed on size mismatch, duplicate names, or any missing required
 * name. On success, *ordered holds one pointer per required info (session
 * order), independent of the caller's vector order.
 */
inline StatusCode match_required_inputs(const std::vector<TensorInfo>& required,
                                        const std::vector<NamedTensor>& inputs,
                                        std::vector<const NamedTensor*>* ordered,
                                        std::string* err) {
    if (ordered == nullptr) {
        if (err != nullptr) {
            *err = "internal error: ordered input list is null";
        }
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    if (inputs.size() != required.size()) {
        if (err != nullptr) {
            *err = "session expects " + std::to_string(required.size()) + " inputs, got " +
                   std::to_string(inputs.size());
        }
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = i + 1; j < inputs.size(); ++j) {
            if (inputs[i].name == inputs[j].name) {
                if (err != nullptr) {
                    *err = "duplicate input tensor: " + inputs[i].name;
                }
                return StatusCode::MODEL_RUN_SESSION_FAILED;
            }
        }
    }
    ordered->clear();
    ordered->reserve(required.size());
    for (const auto& info : required) {
        const auto found = std::find_if(
            inputs.begin(), inputs.end(),
            [&info](const NamedTensor& named) { return named.name == info.name; });
        if (found == inputs.end()) {
            if (err != nullptr) {
                *err = "missing input tensor: " + info.name;
            }
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        ordered->push_back(&(*found));
    }
    return StatusCode::OK;
}

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_SESSION_IO_H
