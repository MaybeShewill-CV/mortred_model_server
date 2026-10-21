/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.h
 * Date: 24-3-13
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H

#include <map>
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

template <typename INPUT, typename OUTPUT> class YoloV8Detector : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    YoloV8Detector();
    ~YoloV8Detector() override = default;

    YoloV8Detector(const YoloV8Detector &transformer) = delete;
    YoloV8Detector &operator=(const YoloV8Detector &transformer) = delete;

  protected:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    DetectionParams _m_detection_params;
    // host input node size (network space)
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace object_detection
} // namespace models
} // namespace mortred

#include "yolov8_detector.inl"

#endif // MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H
