/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: bisenetv2.h
* Date: 22-6-9
************************************************/

#ifndef MM_AI_SERVER_BISENETV2_H
#define MM_AI_SERVER_BISENETV2_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace mortred {
namespace models {
namespace scene_segmentation {

template<typename INPUT, typename OUTPUT>
class BiseNetV2 : public mortred::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * 构造函数
    * @param config
    */
    BiseNetV2();

    /***
     *
     */
    ~BiseNetV2() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    BiseNetV2(const BiseNetV2& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    BiseNetV2& operator=(const BiseNetV2& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    mortred::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    mortred::common::StatusCode run(const INPUT& input, OUTPUT& output) override;


    /***
     * if db text detector successfully initialized
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

#endif //MM_AI_SERVER_BISENETV2_H
