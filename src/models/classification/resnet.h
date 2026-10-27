/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: resnet.h
 * Date: 22-6-14
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_RESNET_H
#define MORTRED_MODEL_SERVER_RESNET_H

#include <string>
#include <unordered_map>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace classification {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class ResNet : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    ResNet();
    ~ResNet() override = default;

    ResNet(const ResNet &transformer) = delete;
    ResNet &operator=(const ResNet &transformer) = delete;

  protected:
    // image -> normalized CV_32FC3 HWC mat of _m_input_tensor_size (batch share)
    cv::Mat preprocess_mat(const cv::Mat &image);

    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    std::unordered_map<uint16_t, std::string> _m_class_id2names;
    cv::Size _m_input_tensor_size = cv::Size(224, 224);
};

} // namespace classification
} // namespace models
} // namespace mortred

#include "resnet.inl"

#endif // MORTRED_MODEL_SERVER_RESNET_H
