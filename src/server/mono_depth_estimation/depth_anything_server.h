/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: DepthAnythingServer.h
 * Date: 24-1-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DEPTH_ANYTHING_SERVER_H
#define MORTRED_MODEL_SERVER_DEPTH_ANYTHING_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace mono_depth_estimation {
class DepthAnythingServer : public jinq::server::BaseAiServer {
  public:
    
    /***
    * constructor
    * @param config
     */
    DepthAnythingServer();

    /***
     *
     */
    ~DepthAnythingServer() override;

    /***
    * constructor
    * @param transformer
     */
    DepthAnythingServer(const DepthAnythingServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    DepthAnythingServer& operator=(const DepthAnythingServer& transformer) = delete;

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

#endif // MORTRED_MODEL_SERVER_DEPTH_ANYTHING_SERVER_H
