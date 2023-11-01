/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: metric3d_server.h
 * Date: 23-11-1
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_METRIC3D_SERVER_H
#define MORTRED_MODEL_SERVER_METRIC3D_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace mono_depth_estimation {
class Metric3DServer : public jinq::server::BaseAiServer {
  public:

    /***
    * constructor
    * @param config
     */
    Metric3DServer();

    /***
     *
     */
    ~Metric3DServer() override;

    /***
    * constructor
    * @param transformer
     */
    Metric3DServer(const Metric3DServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    Metric3DServer& operator=(const Metric3DServer& transformer) = delete;

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

#endif // MORTRED_MODEL_SERVER_METRIC3D_SERVER_H
