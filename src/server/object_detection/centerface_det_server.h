/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: centerface_det_server.h
 * Date: 23-10-18
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_CENTERFACE_DET_SERVER_H
#define MORTRED_MODEL_SERVER_CENTERFACE_DET_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace object_detection {
class CenterfaceDetServer : public jinq::server::BaseAiServer {
  public:

    /***
    * constructor
    * @param config
     */
    CenterfaceDetServer();

    /***
     *
     */
    ~CenterfaceDetServer() override;

    /***
    * constructor
    * @param transformer
     */
    CenterfaceDetServer(const CenterfaceDetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    CenterfaceDetServer& operator=(const CenterfaceDetServer& transformer) = delete;

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

#endif // MORTRED_MODEL_SERVER_CENTERFACE_DET_SERVER_H
