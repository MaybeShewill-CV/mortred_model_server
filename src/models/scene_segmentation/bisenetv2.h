/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: bisenetv2.h
* Date: 22-6-9
************************************************/

#ifndef MORTRED_MODEL_SERVER_BISENETV2_H
#define MORTRED_MODEL_SERVER_BISENETV2_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

template<typename INPUT, typename OUTPUT>
class BiseNetV2 : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    * @param config
    */
    BiseNetV2();

    /***
     *
     */
    ~BiseNetV2() override;

    /***
    * constructor
    * @param transformer
    */
    BiseNetV2(const BiseNetV2& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    BiseNetV2& operator=(const BiseNetV2& transformer) = delete;

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
     * if model successfully initialized
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

#include "bisenetv2.inl"

#endif //MORTRED_MODEL_SERVER_BISENETV2_H
