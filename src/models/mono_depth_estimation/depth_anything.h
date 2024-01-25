/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: depth_anything.h
 * Date: 24-1-25
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DEPTH_ANYTHING_H
#define MORTRED_MODEL_SERVER_DEPTH_ANYTHING_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace mono_depth_estimation {

template <typename INPUT, typename OUTPUT>
class DepthAnything : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    DepthAnything();

    /***
     *
     */
    ~DepthAnything() override;

    /***
     * constructor
     * @param transformer
     */
    DepthAnything(const DepthAnything &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DepthAnything &operator=(const DepthAnything &transformer) = delete;

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
} // namespace mono_depth_estimation
} // namespace models
} // namespace jinq

#include "depth_anything.inl"

#endif // MORTRED_MODEL_SERVER_DEPTH_ANYTHING_H
