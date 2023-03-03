/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov6_det_server.h
* Date: 23-3-3
************************************************/

#ifndef MM_AI_SERVER_YOLOV6DETSERVER_H
#define MM_AI_SERVER_YOLOV6DETSERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace object_detection {
class YoloV6DetServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    YoloV6DetServer();

    /***
     *
     */
    ~YoloV6DetServer() override;

    /***
    * constructor
    * @param transformer
    */
    YoloV6DetServer(const YoloV6DetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV6DetServer& operator=(const YoloV6DetServer& transformer) = delete;

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

#endif //MM_AI_SERVER_YOLOV6DETSERVER_H
