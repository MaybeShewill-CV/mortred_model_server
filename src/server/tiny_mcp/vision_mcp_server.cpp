/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: VisionMcpServer.cpp
 * Date: 25-4-9
 ************************************************/

#include "vision_mcp_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"

#include "common/status_code.h"
#include "common/file_path_util.h"

namespace jinq {
namespace server {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;

namespace tiny_mcp {

/************ Impl Declaration ************/

class VisionMcpServer::Impl {
  public:
    /***
    *
    * @param cfg_file_path
    * @return
     */
    StatusCode init(const decltype(toml::parse(""))& config);

    /***
     *
     */
    void run() {
        _m_server->start(_m_host.c_str(), _m_port);
    }

    /***
     *
     * @param task
     */
    void serve_process(WFHttpTask* task);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    }

  private:
    // server
    std::unique_ptr<WFHttpServer> _m_server;
    // host
    std::string _m_host;
    // port
    int16_t _m_port = 0;
    // base uri
    std::string _m_base_uri;
    // init flag
    bool _m_successfully_initialized = false;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode VisionMcpServer::Impl::init(const decltype(toml::parse("")) &config) {
    if (!config.contains("MCP_SERVER")) {
        LOG(ERROR) << "config file missing \'MCP_SERVER\' field";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    // init params
    const auto& mcp_cfg = config.at("MCP_SERVER");
    _m_host = mcp_cfg.at("host").as_string();
    _m_port = static_cast<int16_t >(mcp_cfg.at("port").as_integer());
    _m_base_uri = mcp_cfg.at("base_server_uri").as_string();

    // bind server process
    auto&& proc = [&](auto arg) { return serve_process(arg); };
    _m_server = std::make_unique<WFHttpServer>(proc);

    _m_successfully_initialized = true;
    LOG(INFO) << "mcp server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param config
 * @return
 */
void VisionMcpServer::Impl::serve_process(WFHttpTask *task) {
    // welcome message
    if (strcmp(task->get_req()->get_request_uri(), (_m_base_uri + "/welcome").c_str()) == 0) {
        task->get_resp()->append_output_body("<html>Welcome to jinq ai server</html>");
        return;
    }
    //
}

/***
 *
 */
VisionMcpServer::VisionMcpServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
VisionMcpServer::~VisionMcpServer() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode VisionMcpServer::init(const decltype(toml::parse("")) &config) {
    return _m_impl->init(config);
}

/***
 *
 */
void VisionMcpServer::run() {
   return _m_impl->run();
}

/***
 *
 * @return
 */
bool VisionMcpServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}

}
}
}