/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: VisionMcpServer.h
 * Date: 25-4-9
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_VISION_MCP_SERVER_H
#define MORTRED_MODEL_SERVER_VISION_MCP_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace tiny_mcp {
class VisionMcpServer {
  public:

    /***
    * constructor
    * @param config
     */
    VisionMcpServer();

    /***
     *
     */
    ~VisionMcpServer();

    /***
    * constructor
    * @param transformer
     */
    VisionMcpServer(const VisionMcpServer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    VisionMcpServer& operator=(const VisionMcpServer& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     */
    void run();

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_impl;
};

}
}
}

#endif // MORTRED_MODEL_SERVER_VISION_MCP_SERVER_H
