/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: Yolov5DetServer.h
* Date: 22-6-22
************************************************/

#ifndef MM_AI_SERVER_YOLOV5DETSERVER_H
#define MM_AI_SERVER_YOLOV5DETSERVER_H

#include <memory>

#include "server/base_server.h"

namespace mortred {
namespace server {
namespace object_detection {
class YoloV5DetServer : public mortred::server::BaseAiServer {
public:

    /***
    * 构造函数
    * @param config
    */
    YoloV5DetServer();

    /***
     *
     */
    ~YoloV5DetServer() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    YoloV5DetServer(const YoloV5DetServer& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    YoloV5DetServer& operator=(const YoloV5DetServer& transformer) = delete;

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

#endif //MM_AI_SERVER_YOLOV5DETSERVER_H
