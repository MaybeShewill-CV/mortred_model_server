/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2.h
* Date: 22-6-13
************************************************/

#ifndef MORTRED_MODEL_SERVER_MOBILENETV2_H
#define MORTRED_MODEL_SERVER_MOBILENETV2_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace classification {

template<typename INPUT, typename OUTPUT>
class MobileNetv2 : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    * @param config
    */
    MobileNetv2();

    /***
     *
     */
    ~MobileNetv2() override;

    /***
    * constructor
    * @param transformer
    */
    MobileNetv2(const MobileNetv2& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    MobileNetv2& operator=(const MobileNetv2& transformer) = delete;

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
     * if mobilenetv2 model successfully initialized
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

#include "mobilenetv2.inl"

#endif //MORTRED_MODEL_SERVER_MOBILENETV2_H
