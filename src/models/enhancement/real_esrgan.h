/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: RealEsrGan.h
* Date: 22-9-29
************************************************/

#ifndef MORTRED_MODEL_SERVER_REALESRGAN_H
#define MORTRED_MODEL_SERVER_REALESRGAN_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace enhancement {

template <typename INPUT, typename OUTPUT>
class RealEsrGan : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:
    /***
     * construct function
     * @param config
     */
    RealEsrGan();

    /***
     *
     */
    ~RealEsrGan() override;

    /***
     * construct function
     * @param transformer
     */
    RealEsrGan(const RealEsrGan& transformer) = delete;

    /***
     * construct function
     * @param transformer
     * @return
     */
    RealEsrGan& operator=(const RealEsrGan& transformer) = delete;

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
     * model init flag
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

#include "real_esrgan.inl"

#endif //MORTRED_MODEL_SERVER_REALESRGAN_H
