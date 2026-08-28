/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ddpm_unet.h
 * Date: 24-4-23
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DDPMUNET_H
#define MORTRED_MODEL_SERVER_DDPMUNET_H

#include <vector>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace diffusion {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT> class DDPMUNet : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    DDPMUNet();
    ~DDPMUNet() override = default;

    DDPMUNet(const DDPMUNet &transformer) = delete;
    DDPMUNet &operator=(const DDPMUNet &transformer) = delete;

  private:
    jinq::models::backend::PreparedInput prepare_inputs(const INPUT &input) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    // concrete xt input shape parsed from the session io ([N,C,H,W])
    std::vector<int64_t> _m_xt_shape;
};

} // namespace diffusion
} // namespace models
} // namespace jinq

#include "ddpm_unet.inl"

#endif // MORTRED_MODEL_SERVER_DDPMUNET_H
