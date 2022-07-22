/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: pphuman_seg_server.h
* Date: 22-7-22
************************************************/

#ifndef MM_AI_SERVER_PPHUMANSEGSERVER_H
#define MM_AI_SERVER_PPHUMANSEGSERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace scene_segmentation {
class PPHumanSegServer : public jinq::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    PPHumanSegServer();

    /***
     *
     */
    ~PPHumanSegServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    PPHumanSegServer(const PPHumanSegServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    PPHumanSegServer& operator=(const PPHumanSegServer& transformer) = delete;

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

#endif //MM_AI_SERVER_PPHUMANSEGSERVER_H
