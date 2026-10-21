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

namespace mortred {
namespace models {
namespace enhancement {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class AttentiveGanDerain : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    AttentiveGanDerain();
    ~AttentiveGanDerain() override = default;

    AttentiveGanDerain(const AttentiveGanDerain &transformer) = delete;
    AttentiveGanDerain &operator=(const AttentiveGanDerain &transformer) = delete;

  protected:
    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

  private:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode on_init(const toml::table &params) override;

    cv::Size _m_input_size_host;
};

} // namespace enhancement
} // namespace models
} // namespace mortred

#include "attentive_gan_derain_net.inl"

#endif // MORTRED_MODEL_SERVER_ATTENTIVE_GAN_DERAIN_NET_H
