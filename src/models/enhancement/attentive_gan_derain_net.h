/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: attentive_gan_derain_net.h
* Date: 22-6-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_ATTENTIVE_GAN_DERAIN_NET_H
#define MORTRED_MODEL_SERVER_ATTENTIVE_GAN_DERAIN_NET_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace enhancement {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT>
class AttentiveGanDerain : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    AttentiveGanDerain();
    ~AttentiveGanDerain() override = default;

    AttentiveGanDerain(const AttentiveGanDerain& transformer) = delete;
    AttentiveGanDerain& operator=(const AttentiveGanDerain& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode on_init(const toml::table& params) override;

    cv::Size _m_input_size_user;
    cv::Size _m_input_size_host;
};

} // namespace enhancement
} // namespace models
} // namespace jinq

#include "attentive_gan_derain_net.inl"

#endif // MORTRED_MODEL_SERVER_ATTENTIVE_GAN_DERAIN_NET_H
