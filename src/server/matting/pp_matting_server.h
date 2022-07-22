/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: pp_matting_server.h
* Date: 22-7-22
************************************************/

#ifndef MM_AI_SERVER_PPMATTINGSERVER_H
#define MM_AI_SERVER_PPMATTINGSERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace matting {
class PPMattingServer : public jinq::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    PPMattingServer();

    /***
     *
     */
    ~PPMattingServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    PPMattingServer(const PPMattingServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    PPMattingServer& operator=(const PPMattingServer& transformer) = delete;

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

#endif //MM_AI_SERVER_PPMATTINGSERVER_H
