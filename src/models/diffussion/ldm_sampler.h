/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ldm_sampler.h
 * Date: 24-5-24
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LDM_SAMPLER_H
#define MORTRED_MODEL_SERVER_LDM_SAMPLER_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace diffusion {

template <typename INPUT, typename OUTPUT>
class LDMSampler : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    LDMSampler();

    /***
     *
     */
    ~LDMSampler() override;

    /***
     * constructor
     * @param transformer
     */
    LDMSampler(const LDMSampler &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    LDMSampler &operator=(const LDMSampler &transformer) = delete;

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

#include "ldm_sampler.inl"

#endif // MORTRED_MODEL_SERVER_LDM_SAMPLER_H
