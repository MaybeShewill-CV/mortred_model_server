/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: Yolov5DetServer.h
* Date: 22-6-22
************************************************/

#ifndef MM_AI_SERVER_YOLOV5DETSERVER_H
#define MM_AI_SERVER_YOLOV5DETSERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace object_detection {
class YoloV5DetServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    YoloV5DetServer();

    /***
     *
     */
    ~YoloV5DetServer() override;

    /***
    * constructor
    * @param transformer
    */
    YoloV5DetServer(const YoloV5DetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV5DetServer& operator=(const YoloV5DetServer& transformer) = delete;

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

#endif //MM_AI_SERVER_YOLOV5DETSERVER_H
