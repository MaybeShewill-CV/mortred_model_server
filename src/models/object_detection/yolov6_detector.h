/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov6_detector.h
 * Date: 23-3-3
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H

#include <map>
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

template <typename INPUT, typename OUTPUT> class YoloV6Detector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    YoloV6Detector();
    ~YoloV6Detector() override = default;

    YoloV6Detector(const YoloV6Detector &transformer) = delete;
    YoloV6Detector &operator=(const YoloV6Detector &transformer) = delete;

  protected:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    DetectionParams _m_detection_params;
    // input node size
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace object_detection
} // namespace models
} // namespace jinq

#include "yolov6_detector.inl"

#endif // MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H
