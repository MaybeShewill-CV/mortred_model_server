/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: bisenetv2.h
 * Date: 22-6-9
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_BISENETV2_H
#define MORTRED_MODEL_SERVER_BISENETV2_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace scene_segmentation {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT> class BiseNetV2 : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    BiseNetV2();
    ~BiseNetV2() override = default;

    BiseNetV2(const BiseNetV2 &transformer) = delete;
    BiseNetV2 &operator=(const BiseNetV2 &transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    // model input tensor size
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace scene_segmentation
} // namespace models
} // namespace jinq

#include "bisenetv2.inl"

#endif // MORTRED_MODEL_SERVER_BISENETV2_H
