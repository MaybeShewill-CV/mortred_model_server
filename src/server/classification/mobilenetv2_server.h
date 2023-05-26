/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2_server.h
* Date: 22-6-21
************************************************/

#ifndef MM_AI_SERVER_MOBILENETV2_SERVER_H
#define MM_AI_SERVER_MOBILENETV2_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace classification {
class MobileNetv2Server : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    MobileNetv2Server();

    /***
     * constructor
     */
    ~MobileNetv2Server() override;

    /***
    * constructor
    * @param transformer
    */
    MobileNetv2Server(const MobileNetv2Server& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    MobileNetv2Server& operator=(const MobileNetv2Server& transformer) = delete;

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
     * init flag
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
