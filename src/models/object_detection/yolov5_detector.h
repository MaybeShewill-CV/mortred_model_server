/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov5_detector.h
 * Date: 22-6-7
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV5_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV5_DETECTOR_H

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

template <typename INPUT, typename OUTPUT> class YoloV5Detector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    YoloV5Detector();
    ~YoloV5Detector() override = default;

    YoloV5Detector(const YoloV5Detector &transformer) = delete;
    YoloV5Detector &operator=(const YoloV5Detector &transformer) = delete;

  protected:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    DetectionParams _m_detection_params;
    // input node size
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace object_detection
} // namespace models
} // namespace jinq

#include "yolov5_detector.inl"

#endif // MORTRED_MODEL_SERVER_YOLOV5_DETECTOR_H
