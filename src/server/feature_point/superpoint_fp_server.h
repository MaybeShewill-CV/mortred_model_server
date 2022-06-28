/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: superpoint_fp_server.h
* Date: 22-6-29
************************************************/

#ifndef MM_AI_SERVER_SUPERPOINTFPSERVER_H
#define MM_AI_SERVER_SUPERPOINTFPSERVER_H

#include <memory>

#include "server/base_server.h"

namespace morted {
namespace server {
namespace feature_point {
class SuperpointFpServer : public morted::server::BaseAiServer {
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

#endif //MM_AI_SERVER_SUPERPOINTFPSERVER_H
