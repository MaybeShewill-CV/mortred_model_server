/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: resnet_server_ins.h
* Date: 22-6-21
************************************************/

#ifndef MM_AI_SERVER_RESNET_SERVER_H
#define MM_AI_SERVER_RESNET_SERVER_H

#include <memory>

#include "server/base_server.h"

namespace morted {
namespace server {
namespace classification {
class ResNetServer : public morted::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    ResNetServer();

    /***
     *
     */
    ~ResNetServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    ResNetServer(const ResNetServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    ResNetServer& operator=(const ResNetServer& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    morted::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

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

#endif //MM_AI_SERVER_RESNET_SERVER_H
