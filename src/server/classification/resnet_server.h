/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: resnet_server.h
* Date: 22-6-21
************************************************/

#ifndef MM_AI_SERVER_RESNET_SERVER_H
#define MM_AI_SERVER_RESNET_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace classification {
class ResNetServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    ResNetServer();

    /***
     *
     */
    ~ResNetServer() override;

    /***
    * constructor
    * @param transformer
    */
    ResNetServer(const ResNetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    ResNetServer& operator=(const ResNetServer& transformer) = delete;

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

#endif //MM_AI_SERVER_RESNET_SERVER_H
