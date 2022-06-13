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

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace morted {
namespace models {
namespace enhancement {

template<typename INPUT, typename OUTPUT>
class EnlightenGan : public morted::models::BaseAiModel<INPUT, OUTPUT> {
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
    EnlightenGan(const EnlightenGan& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    EnlightenGan& operator=(const EnlightenGan& transformer) = delete;

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
    morted::common::StatusCode run(const INPUT& input, std::vector<OUTPUT>& output) override;


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

#include "enlightengan.inl"

#endif //MM_AI_SERVER_ENLIGHTENGAN_H
