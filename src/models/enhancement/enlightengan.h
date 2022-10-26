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

namespace jinq {
namespace models {
namespace enhancement {

template <typename INPUT, typename OUTPUT> 
class EnlightenGan : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
     * construct function
     * @param config
     */
    EnlightenGan();

    /***
     *
     */
    ~EnlightenGan() override;

    /***
     * construct function
     * @param transformer
     */
    EnlightenGan(const EnlightenGan &transformer) = delete;

    /***
     * construct function
     * @param transformer
     * @return
     */
    EnlightenGan &operator=(const EnlightenGan &transformer) = delete;

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

#include "enlightengan.inl"

#endif // MM_AI_SERVER_ENLIGHTENGAN_H
