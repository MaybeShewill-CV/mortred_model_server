/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: bisenetv2_server.h
* Date: 22-7-04
************************************************/

#ifndef MORTRED_MODEL_SERVER_BISENETV2_SERVER_H
#define MORTRED_MODEL_SERVER_BISENETV2_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace scene_segmentation {
class BiseNetV2Server : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    BiseNetV2Server();

    /***
     *
     */
    ~BiseNetV2Server() override;

    /***
    * constructor
    * @param transformer
    */
    BiseNetV2Server(const BiseNetV2Server& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    BiseNetV2Server& operator=(const BiseNetV2Server& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param task
     */
    void serve_process(WFHttpTask* task) override;

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const override;

private:
    class Impl;
    std::unique_ptr<Impl> _m_impl;
};
}
}
}

#endif //MORTRED_MODEL_SERVER_BISENETV2_SERVER_H
