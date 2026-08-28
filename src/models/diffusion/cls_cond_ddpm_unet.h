/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cls_cond_ddpm_unet.h
 * Date: 26-8-17
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_CLS_COND_DDPM_UNET_H
#define MORTRED_MODEL_SERVER_CLS_COND_DDPM_UNET_H

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

template <typename INPUT, typename OUTPUT> class ClsCondDDPMUNet : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    ClsCondDDPMUNet();
    ~ClsCondDDPMUNet() override = default;

    ClsCondDDPMUNet(const ClsCondDDPMUNet &transformer) = delete;
    ClsCondDDPMUNet &operator=(const ClsCondDDPMUNet &transformer) = delete;

  private:
    jinq::models::backend::PreparedInput prepare_inputs(const INPUT &input) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;
};

} // namespace diffusion
} // namespace models
} // namespace jinq

#include "cls_cond_ddpm_unet.inl"

#endif // MORTRED_MODEL_SERVER_CLS_COND_DDPM_UNET_H
