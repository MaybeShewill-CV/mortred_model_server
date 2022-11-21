/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: modnet_server.h
* Date: 22-7-22
************************************************/

#ifndef MM_AI_SERVER_MODNETSERVER_H
#define MM_AI_SERVER_MODNETSERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace matting {
class ModNetServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    ModNetServer();

    /***
     *
     */
    ~ModNetServer() override;

    /***
    * constructor
    * @param transformer
    */
    ModNetServer(const ModNetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    ModNetServer& operator=(const ModNetServer& transformer) = delete;

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

#endif //MM_AI_SERVER_MODNETSERVER_H
