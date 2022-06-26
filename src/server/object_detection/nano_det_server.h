/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: nanodet_server.h
* Date: 22-6-22
************************************************/

#ifndef MM_AI_SERVER_NANODET_SERVER_H
#define MM_AI_SERVER_NANODET_SERVER_H

#include <memory>

#include "server/base_server.h"

namespace morted {
namespace server {
namespace object_detection {
class NanoDetServer : public morted::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    NanoDetServer();

    /***
     *
     */
    ~NanoDetServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    NanoDetServer(const NanoDetServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    NanoDetServer& operator=(const NanoDetServer& transformer) = delete;

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

#endif //MM_AI_SERVER_NANODET_SERVER_H