/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: lightglue.h
* Date: 23-11-03
************************************************/

#ifndef MORTRED_MODEL_SERVER_LIGHTGLUE_H
#define MORTRED_MODEL_SERVER_LIGHTGLUE_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace feature_point {

template <typename INPUT, typename OUTPUT>
class LightGlue : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    LightGlue();

    /***
     *
     */
    ~LightGlue() override;

    /***
     * constructor
     * @param transformer
     */
    LightGlue(const LightGlue &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    LightGlue &operator=(const LightGlue &transformer) = delete;

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
    jinq::common::StatusCode run(const INPUT& input, OUTPUT& output) override;

    /***
     * if lightglue matcher successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

} // namespace feature_point
} // namespace models
} // namespace jinq

#include "lightglue.inl"

#endif // MORTRED_MODEL_SERVER_LIGHTGLUE_H