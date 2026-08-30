/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: rtdetr_detector.h
 * Date: 2026-08-30
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_RTDETR_DETECTOR_H
#define MORTRED_MODEL_SERVER_RTDETR_DETECTOR_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"
#include "models/io/object_detection.h"

namespace jinq {
namespace models {
namespace object_detection {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT> class RtdetrDetector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    RtdetrDetector();
    ~RtdetrDetector() override = default;

    RtdetrDetector(const RtdetrDetector &transformer) = delete;
    RtdetrDetector &operator=(const RtdetrDetector &transformer) = delete;

  protected:
    // request image -> named input tensors
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    // named output tensors + request geometry -> task output
    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext &context, OUTPUT &output) override;

    // model specific keys from [RTDETR.params]
    StatusCode on_init(const toml::table &params) override;
};

} // namespace object_detection
} // namespace models
} // namespace jinq

#include "rtdetr_detector.inl"

#endif // MORTRED_MODEL_SERVER_RTDETR_DETECTOR_H
