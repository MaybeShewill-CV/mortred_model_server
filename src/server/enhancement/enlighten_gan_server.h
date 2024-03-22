/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: enlighten_gan_server.h
* Date: 22-7-04
************************************************/

#ifndef MORTRED_MODEL_SERVER_ENLIGHTEN_GAN_SERVER_H
#define MORTRED_MODEL_SERVER_ENLIGHTEN_GAN_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace enhancement {
class EnlightenGanServer : public jinq::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    EnlightenGanServer();

    /***
     *
     */
    ~EnlightenGanServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    EnlightenGanServer(const EnlightenGanServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    EnlightenGanServer& operator=(const EnlightenGanServer& transformer) = delete;

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

#endif //MORTRED_MODEL_SERVER_ENLIGHTEN_GAN_SERVER_H
