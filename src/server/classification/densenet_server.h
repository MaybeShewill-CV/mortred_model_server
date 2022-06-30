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

namespace mortred {
namespace server {
namespace classification {
class DenseNetServer : public mortred::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    DenseNetServer();

    /***
     *
     */
    ~DenseNetServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    DenseNetServer(const DenseNetServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    DenseNetServer& operator=(const DenseNetServer& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    mortred::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

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
