#ifndef MORTRED_MODELS_BACKEND_F32_OUTPUT_H
#define MORTRED_MODELS_BACKEND_F32_OUTPUT_H

#include <string>
#include <vector>

#include <glog/logging.h>

#include "common/status_code.h"
#include "models/backend/tensor_contract.h"

namespace jinq {
namespace models {
namespace backend {

using jinq::common::StatusCode;

/*** typed view of an f32 output which passed its complete tensor contract ***/
struct F32OutputView {
    const Tensor *tensor = nullptr;
    const float *data = nullptr;
};

/***
 * Find, validate and expose a named f32 output. Malformed backend output is
 * rejected at the model boundary instead of reaching task-specific decoding.
 */
inline StatusCode validated_f32_named_output(const std::vector<NamedTensor> &outputs, const std::string &name,
                                             const TensorContract &contract, const std::string &log_prefix, F32OutputView *view = nullptr) {
    const auto *named = find_output(outputs, name);
    if (named == nullptr) {
        LOG(ERROR) << log_prefix << " output tensor '" << name << "' is missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    std::string error;
    if (!validate_output_tensor(*named, contract, &error)) {
        LOG(ERROR) << log_prefix << " output contract failed: " << error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }

    const float *data = nullptr;
    if (!get_f32_data(named->tensor, &data, &error) ||
        !require_finite_f32(data, static_cast<size_t>(named->tensor.element_count()), named->name, &error)) {
        LOG(ERROR) << log_prefix << " output contract failed: " << error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    if (view != nullptr) {
        view->tensor = &named->tensor;
        view->data = data;
    }
    return StatusCode::OK;
}

/*** single-output models do not need to know engine-generated output names ***/
inline StatusCode validated_f32_first_output(const std::vector<NamedTensor> &outputs, const TensorContract &contract,
                                             const std::string &log_prefix, F32OutputView *view = nullptr) {
    if (outputs.empty()) {
        LOG(ERROR) << log_prefix << " output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    return validated_f32_named_output(outputs, outputs.front().name, contract, log_prefix, view);
}

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_F32_OUTPUT_H
