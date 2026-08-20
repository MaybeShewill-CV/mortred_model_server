/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: autoencoder_kl.h
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_AUTOENCODER_KL_H
#define MORTRED_MODEL_SERVER_AUTOENCODER_KL_H

#include <vector>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace diffusion {

template <typename INPUT, typename OUTPUT>
class AutoEncoderKL : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    AutoEncoderKL();
    ~AutoEncoderKL() override = default;

    AutoEncoderKL(const AutoEncoderKL& transformer) = delete;
    AutoEncoderKL& operator=(const AutoEncoderKL& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> make_inputs(const INPUT& input) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;
};

} // namespace diffusion
} // namespace models
} // namespace jinq

#include "autoencoder_kl.inl"

#endif //MORTRED_MODEL_SERVER_AUTOENCODER_KL_H
