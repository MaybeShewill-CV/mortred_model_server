#ifndef MORTRED_MODELS_BACKEND_INFERENCE_CONTEXT_H
#define MORTRED_MODELS_BACKEND_INFERENCE_CONTEXT_H

#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "common/status_code.h"
#include "models/backend/param_spec.h"
#include "models/backend/tensor.h"

namespace jinq {
namespace models {
namespace backend {

/*** request-scoped geometry carried from preprocessing to postprocessing */
struct InferenceContext {
    cv::Size source_size;
    cv::Size network_size;

    /***
     * the exact cv::Mat that produced the input tensors (refcounted, no pixel
     * copy). Postprocess passes that need source pixels (e.g. restoring an
     * alpha channel) must read them from here instead of stashing per-request
     * state in model members: one model instance serves many requests and the
     * batch path interleaves pre/postprocess calls across items.
     */
    cv::Mat source_image;

    /***
     * request-scoped parameter view travelling from prepare_inputs to
     * postprocess. nullptr on the legacy single-image path, so every
     * get_f32(key, config_default) call stays on its config default.
     */
    const ParamSet *params = nullptr;
};

/*** all data produced by preprocessing for exactly one inference request */
struct PreparedInput {
    std::vector<backend::NamedTensor> inputs;
    InferenceContext context;
    jinq::common::StatusCode status = jinq::common::StatusCode::OK;
    std::string error;

    static PreparedInput invalid(jinq::common::StatusCode status, std::string error) {
        PreparedInput result;
        result.status = status;
        result.error = std::move(error);
        return result;
    }
};

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_INFERENCE_CONTEXT_H
