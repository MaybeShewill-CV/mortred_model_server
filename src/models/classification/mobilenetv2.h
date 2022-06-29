/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2.h
* Date: 22-6-13
************************************************/

#ifndef MM_AI_SERVER_MOBILENETV2_H
#define MM_AI_SERVER_MOBILENETV2_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace mortred {
namespace models {
namespace classification {

template<typename INPUT, typename OUTPUT>
class MobileNetv2 : public mortred::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * 构造函数
    * @param config
    */
    MobileNetv2();

    /***
     *
     */
    ~MobileNetv2() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    MobileNetv2(const MobileNetv2& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    MobileNetv2& operator=(const MobileNetv2& transformer) = delete;

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

#include "mobilenetv2.inl"

#endif //MM_AI_SERVER_MOBILENETV2_H
