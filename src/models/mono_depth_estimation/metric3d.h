/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: Metric3d.h
 * Date: 23-10-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_METRIC3D_H
#define MORTRED_MODEL_SERVER_METRIC3D_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace mono_depth_estimation {

template <typename INPUT, typename OUTPUT> 
class Metric3D : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * constructor
     * @param config
     */
    Metric3D();

    /***
     *
     */
    ~Metric3D() override;

    /***
     * constructor
     * @param transformer
     */
    Metric3D(const Metric3D &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    Metric3D &operator=(const Metric3D &transformer) = delete;

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

#include "metric3d.inl"

#endif // MORTRED_MODEL_SERVER_METRIC3D_H
