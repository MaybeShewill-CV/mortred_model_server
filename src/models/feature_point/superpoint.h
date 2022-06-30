/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: superpoint.h
* Date: 22-6-15
************************************************/

#ifndef MMAISERVER_SUPERPOINT_H
#define MMAISERVER_SUPERPOINT_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace feature_point {

template <typename INPUT, typename OUTPUT>
class SuperPoint : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * 构造函数
     * @param config
     */
    SuperPoint();

    /***
     *
     */
    ~SuperPoint() override;

    /***
     * 赋值构造函数
     * @param transformer
     */
    SuperPoint(const SuperPoint &transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    SuperPoint &operator=(const SuperPoint &transformer) = delete;

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
     * if db text detector successfully initialized
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

#include "superpoint.inl"

#endif // MMAISERVER_SUPERPOINT_H