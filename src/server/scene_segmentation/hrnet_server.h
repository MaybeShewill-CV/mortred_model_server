/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: hrnet_server.h
 * Date: 24-1-30
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_HRNET_SERVER_H
#define MORTRED_MODEL_SERVER_HRNET_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace scene_segmentation {
class HRNetServer : public jinq::server::BaseAiServer {
  public:

    /***
    * constructor
    * @param config
     */
    HRNetServer();

    /***
     *
     */
    ~HRNetServer() override;

    /***
    * constructor
    * @param transformer
     */
    HRNetServer(const HRNetServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    HRNetServer& operator=(const HRNetServer& transformer) = delete;

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

#endif // MORTRED_MODEL_SERVER_HRNET_SERVER_H
