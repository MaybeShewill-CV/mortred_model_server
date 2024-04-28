/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ddim_sampler.h
 * Date: 24-4-28
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DDIM_SAMPLER_H
#define MORTRED_MODEL_SERVER_DDIM_SAMPLER_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace diffusion {

template <typename INPUT, typename OUTPUT>
class DDIMSampler : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    DDIMSampler();

    /***
     *
     */
    ~DDIMSampler() override;

    /***
     * constructor
     * @param transformer
     */
    DDIMSampler(const DDIMSampler &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DDIMSampler &operator=(const DDIMSampler &transformer) = delete;

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

#include "ddim_sampler.inl"

#endif // MORTRED_MODEL_SERVER_DDIM_SAMPLER_H
