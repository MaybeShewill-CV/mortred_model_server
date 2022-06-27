/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2_server.h
* Date: 22-6-21
************************************************/

#ifndef MM_AI_SERVER_MOBILENETV2_SERVER_H
#define MM_AI_SERVER_MOBILENETV2_SERVER_H

#include <memory>

#include "server/base_server.h"

namespace morted {
namespace server {
namespace classification {
class MobileNetv2Server : public morted::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    MobileNetv2Server();

    /***
     *
     */
    ~MobileNetv2Server() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    MobileNetv2Server(const MobileNetv2Server& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    MobileNetv2Server& operator=(const MobileNetv2Server& transformer) = delete;

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

#endif //MM_AI_SERVER_MOBILENETV2_SERVER_H
