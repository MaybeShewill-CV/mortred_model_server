/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: ddpm_sampler.h
* Date: 24-4-23
************************************************/

#ifndef MORTRED_MODEL_SERVER_DDPM_SAMPLER_H
#define MORTRED_MODEL_SERVER_DDPM_SAMPLER_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace diffusion {
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
class DDPMUNet;

using DDPMSamplerDenoiseModel = DDPMUNet<
    jinq::models::io_define::diffusion::std_ddpm_unet_input,
    jinq::models::io_define::diffusion::std_ddpm_unet_output>;

template <typename INPUT, typename OUTPUT>
class DDPMSampler : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    DDPMSampler();

    /***
     * share an already-owned denoise model; composite samplers use this to
     * avoid loading the same engine once per scheduler
     */
    explicit DDPMSampler(std::shared_ptr<DDPMSamplerDenoiseModel> denoise_model);

    /***
     *
     */
    ~DDPMSampler() override;

    /***
     * constructor
     * @param transformer
     */
    DDPMSampler(const DDPMSampler &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DDPMSampler &operator=(const DDPMSampler &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    StatusCode init(const toml::table &cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    StatusCode run_impl(const INPUT&input, OUTPUT &output) override;

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

#include "ddpm_sampler.inl"

#endif // MORTRED_MODEL_SERVER_DDPM_SAMPLER_H
