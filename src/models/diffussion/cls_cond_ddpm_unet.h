/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cls_cond_ddpm_unet.h
 * Date: 24-5-8
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_CLS_COND_DDPM_UNET_H
#define MORTRED_MODEL_SERVER_CLS_COND_DDPM_UNET_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace diffusion {

template <typename INPUT, typename OUTPUT>
class ClsCondDDPMUNet : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    ClsCondDDPMUNet();

    /***
     *
     */
    ~ClsCondDDPMUNet() override;

    /***
     * constructor
     * @param transformer
     */
    ClsCondDDPMUNet(const ClsCondDDPMUNet &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    ClsCondDDPMUNet &operator=(const ClsCondDDPMUNet &transformer) = delete;

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

#include "cls_cond_ddpm_unet.inl"

#endif // MORTRED_MODEL_SERVER_CLS_COND_DDPM_UNET_H
