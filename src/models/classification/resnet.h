/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: resnet.h
* Date: 22-6-14
************************************************/

#ifndef MM_AI_SERVER_RESNET_H
#define MM_AI_SERVER_RESNET_H


#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace morted {
namespace models {
namespace classification {

template<typename INPUT, typename OUTPUT>
class ResNet : public morted::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * 构造函数
    * @param config
    */
    ResNet();

    /***
     *
     */
    ~ResNet() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    ResNet(const ResNet& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    ResNet& operator=(const ResNet& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    morted::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    morted::common::StatusCode run(const INPUT& input, OUTPUT& output) override;


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

#include "resnet.inl"

#endif //MM_AI_SERVER_RESNET_H
