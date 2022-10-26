/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: attentive_gan_derain.h
* Date: 22-6-14
************************************************/

#ifndef MMAISERVER_ATTENTIVEGANDERAINNET_H
#define MMAISERVER_ATTENTIVEGANDERAINNET_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace enhancement {

template <typename INPUT, typename OUTPUT> class AttentiveGanDerain : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * construct function
     * @param config
     */
    AttentiveGanDerain();

    /***
     *
     */
    ~AttentiveGanDerain() override;

    /***
     * construct function
     * @param transformer
     */
    AttentiveGanDerain(const AttentiveGanDerain &transformer) = delete;

    /***
     * construct function
     * @param transformer
     * @return
     */
    AttentiveGanDerain &operator=(const AttentiveGanDerain &transformer) = delete;

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

} // namespace enhancement
} // namespace models
} // namespace jinq

#include "attentive_gan_derain_net.inl"

#endif // MMAISERVER_ATTENTIVEGANDERAINNET_H
