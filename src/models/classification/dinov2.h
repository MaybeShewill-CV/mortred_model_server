/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dinov2.h
 * Date: 23-6-12
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DINOV2_H
#define MORTRED_MODEL_SERVER_DINOV2_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace classification {

template<typename INPUT, typename OUTPUT>
class Dinov2 : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:

    /***
    * constructor
    * @param config
     */
    Dinov2();

    /***
     *
     */
    ~Dinov2() override;

    /***
    * constructor
    * @param transformer
     */
    Dinov2(const Dinov2& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    Dinov2& operator=(const Dinov2& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode run(const INPUT& input, OUTPUT& output) override;


    /***
     * if classifier successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}

#include "dinov2.inl"

#endif // MORTRED_MODEL_SERVER_DINOV2_H
