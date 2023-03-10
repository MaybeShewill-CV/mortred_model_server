/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: msocrnet.h
* Date: 23-3-11
************************************************/

#ifndef MM_AI_SERVER_MSOCRNET_H
#define MM_AI_SERVER_MSOCRNET_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

template<typename INPUT, typename OUTPUT>
class MsOcrNet : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    * @param config
    */
    MsOcrNet();

    /***
     *
     */
    ~MsOcrNet() override;

    /***
    * constructor
    * @param transformer
    */
    MsOcrNet(const MsOcrNet& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    MsOcrNet& operator=(const MsOcrNet& transformer) = delete;

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

#include "msocrnet.inl"

#endif //MM_AI_SERVER_MSOCRNET_H