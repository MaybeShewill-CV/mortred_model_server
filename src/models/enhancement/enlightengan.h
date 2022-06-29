/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: enlightengan.h
 * Date: 22-6-13
 ************************************************/

#ifndef MM_AI_SERVER_ENLIGHTENGAN_H
#define MM_AI_SERVER_ENLIGHTENGAN_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"


namespace mortred {
namespace models {
namespace enhancement {

template <typename INPUT, typename OUTPUT> 
class EnlightenGan : public mortred::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * 构造函数
     * @param config
     */
    EnlightenGan();

    /***
     *
     */
    ~EnlightenGan() override;

    /***
     * 赋值构造函数
     * @param transformer
     */
    EnlightenGan(const EnlightenGan &transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    EnlightenGan &operator=(const EnlightenGan &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    mortred::common::StatusCode init(const decltype(toml::parse("")) &cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    mortred::common::StatusCode run(const INPUT &input, OUTPUT &output) override;

    /***
     * if db text detector successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

} // namespace enhancement
} // namespace models
} // namespace mortred

#include "enlightengan.inl"

#endif // MM_AI_SERVER_ENLIGHTENGAN_H
