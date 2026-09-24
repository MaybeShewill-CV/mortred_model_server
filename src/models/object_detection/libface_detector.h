/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.h
 * Date: 22-6-10
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H
#define MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H

#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/backend/tensor_contract.h"
#include "models/model_io_define.h"
#include "models/object_detection/detection_params.h"

namespace jinq {
namespace models {
namespace object_detection {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT> class LibFaceDetector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    LibFaceDetector();
    ~LibFaceDetector() override = default;

    LibFaceDetector(const LibFaceDetector &transformer) = delete;
    LibFaceDetector &operator=(const LibFaceDetector &transformer) = delete;

  protected:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    DetectionParams _m_detection_params;
    // Optional content size from model_input_image_size. Empty means native
    // source size. Either path is then right/bottom padded to /32. Same size
    // is GpuPreprocessDescriptor::pre_crop_size for Path B.
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace object_detection
} // namespace models
} // namespace jinq

#include "libface_detector.inl"

#endif // MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H
