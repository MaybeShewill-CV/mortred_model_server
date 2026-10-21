/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: centerface_detector.h
 * Date: 23-10-18
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H
#define MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H

#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/backend/tensor_contract.h"
#include "models/model_io_define.h"
#include "models/object_detection/detection_params.h"

namespace mortred {
namespace models {
namespace object_detection {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class CenterFaceDetector : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    CenterFaceDetector();
    ~CenterFaceDetector() override = default;

    CenterFaceDetector(const CenterFaceDetector &transformer) = delete;
    CenterFaceDetector &operator=(const CenterFaceDetector &transformer) = delete;

  protected:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    DetectionParams _m_detection_params;
};

} // namespace object_detection
} // namespace models
} // namespace mortred

#include "centerface_detector.inl"

#endif // MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H
