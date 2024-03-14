/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: YoloV8DetServer.h
 * Date: 24-3-14
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV8_DET_SERVER_H
#define MORTRED_MODEL_SERVER_YOLOV8_DET_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace object_detection {
class YoloV8DetServer : public jinq::server::BaseAiServer {
  public:

    /***
    * constructor
    * @param config
     */
    YoloV8DetServer();

    /***
     *
     */
    ~YoloV8DetServer() override;

    /***
    * constructor
    * @param transformer
     */
    YoloV8DetServer(const YoloV8DetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV8DetServer& operator=(const YoloV8DetServer& transformer) = delete;

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

#endif // MORTRED_MODEL_SERVER_YOLOV8_DET_SERVER_H
