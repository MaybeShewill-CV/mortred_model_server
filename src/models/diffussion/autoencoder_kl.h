/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: autoencoder_kl.h
 * Date: 24-5-23
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_AUTOENCODER_KL_H
#define MORTRED_MODEL_SERVER_AUTOENCODER_KL_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace diffusion {

template <typename INPUT, typename OUTPUT>
class AutoEncoderKL : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    AutoEncoderKL();

    /***
     *
     */
    ~AutoEncoderKL() override;

    /***
     * constructor
     * @param transformer
     */
    AutoEncoderKL(const AutoEncoderKL &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    AutoEncoderKL &operator=(const AutoEncoderKL &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse("")) &cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode run(const INPUT &input, OUTPUT &output) override;

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
} // namespace diffusion
} // namespace models
} // namespace jinq

#include "autoencoder_kl.inl"

#endif // MORTRED_MODEL_SERVER_AUTOENCODER_KL_H
