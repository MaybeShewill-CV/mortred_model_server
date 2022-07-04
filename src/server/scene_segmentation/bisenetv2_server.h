/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: bisenetv2_server.h
* Date: 22-7-04
************************************************/

#ifndef MM_AI_SERVER_BISENETV2SERVER_H
#define MM_AI_SERVER_BISENETV2SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace scene_segmentation {
class BiseNetV2Server : public jinq::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    BiseNetV2Server();

    /***
     *
     */
    ~BiseNetV2Server() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    BiseNetV2Server(const BiseNetV2Server& transformer) = delete;

    /***
     * 复制构造函数
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

#endif //MM_AI_SERVER_BISENETV2SERVER_H
