/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: densenet_server.h
* Date: 22-7-1
************************************************/

#ifndef MM_AI_SERVER_DENSENET_SERVER_H
#define MM_AI_SERVER_DENSENET_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace classification {
class DenseNetServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    DenseNetServer();

    /***
     *
     */
    ~DenseNetServer() override;

    /***
    * constructor
    * @param transformer
    */
    DenseNetServer(const DenseNetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DenseNetServer& operator=(const DenseNetServer& transformer) = delete;

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

#endif //MM_AI_SERVER_DENSENET_SERVER_H
