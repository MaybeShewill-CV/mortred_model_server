/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: densenet.h
* Date: 22-6-14
************************************************/

#ifndef MM_AI_SERVER_DENSENET_H
#define MM_AI_SERVER_DENSENET_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace classification {

template<typename INPUT, typename OUTPUT>
class DenseNet : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    * @param config
    */
    DenseNet();

    /***
     *
     */
    ~DenseNet() override;

    /***
    * constructor
    * @param transformer
    */
    DenseNet(const DenseNet& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DenseNet& operator=(const DenseNet& transformer) = delete;

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
     * if classifier successfully initialized
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

#include "densenet.inl"

#endif //MM_AI_SERVER_DENSENET_H
