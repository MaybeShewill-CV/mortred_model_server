/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: modnet_matting.h
* Date: 22-7-19
************************************************/

#ifndef MM_AI_SERVER_MODNET_MATTING_H
#define MM_AI_SERVER_MODNET_MATTING_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

template<typename INPUT, typename OUTPUT>
class ModNetMatting : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * 构造函数
    * @param config
    */
    ModNetMatting();

    /***
     *
     */
    ~ModNetMatting() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    ModNetMatting(const ModNetMatting& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    ModNetMatting& operator=(const ModNetMatting& transformer) = delete;

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

#include "modnet_matting.inl"

#endif //MM_AI_SERVER_MODNET_MATTING_H
