/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: superpoint_fp_server.h
* Date: 22-6-29
************************************************/

#ifndef MORTRED_MODEL_SERVER_SUPERPOINT_FP_SERVER_H
#define MORTRED_MODEL_SERVER_SUPERPOINT_FP_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace feature_point {
class SuperpointFpServer : public jinq::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    SuperpointFpServer();

    /***
     *
     */
    ~SuperpointFpServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    SuperpointFpServer(const SuperpointFpServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    SuperpointFpServer& operator=(const SuperpointFpServer& transformer) = delete;

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

#endif //MORTRED_MODEL_SERVER_SUPERPOINT_FP_SERVER_H
