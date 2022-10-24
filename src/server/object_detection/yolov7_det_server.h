/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov7_det_server.h
* Date: 22-10-24
************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV7DETSERVER_H
#define MORTRED_MODEL_SERVER_YOLOV7DETSERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace object_detection {
class YoloV7DetServer : public jinq::server::BaseAiServer {
public:

    /***
    * constructor
    * @param config
    */
    YoloV7DetServer();

    /***
     *
     */
    ~YoloV7DetServer() override;

    /***
    * constructor
    * @param transformer
    */
    YoloV7DetServer(const YoloV7DetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV7DetServer& operator=(const YoloV7DetServer& transformer) = delete;

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

#endif //MORTRED_MODEL_SERVER_YOLOV7DETSERVER_H
