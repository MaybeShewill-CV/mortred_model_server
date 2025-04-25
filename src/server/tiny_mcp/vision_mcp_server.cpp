/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: VisionMcpServer.cpp
 * Date: 25-4-9
 ************************************************/

#include "vision_mcp_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "workflow/WFFacilities.h"
#include "workflow/HttpUtil.h"

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
        WFFacilities::WaitGroup wait_group(1);
        if (_m_server->start(_m_host.c_str(), _m_port) == 0) {
            wait_group.wait();
            _m_server->stop();
        } else {
            LOG(ERROR) << "Cannot start server";
            return;
        }
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
    // api tools cfg
    std::string _m_api_tool_cfg_path;
    // init flag
    bool _m_successfully_initialized = false;

  private:
    /***
     *
     * @return
     */
    std::string parse_api_tools_cfg_file();

    /***
     *
     * @param request_content
     * @return
     */
    std::string call_api_process(const std::string& request_content);
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

    // init api tools params
    const auto& api_tool_cfg = config.at("API_TOOLS");
    _m_api_tool_cfg_path = api_tool_cfg.at("api_tools_cfg_path").as_string();

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
    // list tools
    if (strcmp(task->get_req()->get_request_uri(), (_m_base_uri + "/list_tools").c_str()) == 0) {
        auto resp_body = parse_api_tools_cfg_file();
        task->get_resp()->append_output_body(resp_body.c_str());
        return;
    }
    // call api
    if (strcmp(task->get_req()->get_request_uri(), (_m_base_uri + "/call_api").c_str()) == 0) {
        const auto& req = task->get_req();
        const auto& req_content = protocol::HttpUtil::decode_chunked_body(req);
        const auto resp_content = call_api_process(req_content);
        task->get_resp()->append_output_body(resp_content.c_str());
        return;
    }
}

/***
 *
 * @return
 */
std::string VisionMcpServer::Impl::parse_api_tools_cfg_file() {
    if (!FilePathUtil::is_file_exist(_m_api_tool_cfg_path)) {
        LOG(ERROR) << "vision mcp server api tool cfg path: " << _m_api_tool_cfg_path << " not exists";
        return "";
    }
    std::ifstream f_in(_m_api_tool_cfg_path.c_str());
    if (!f_in.is_open()) {
        LOG(ERROR) << "vision mcp server api tool cfg path: " << _m_api_tool_cfg_path << " can not be opened";
        return "";
    }

    std::stringstream buffer;
    buffer << f_in.rdbuf();
    std::string json_str = buffer.str();

    return json_str;
}

/***
 *
 * @param request_content
 * @return
 */
std::string VisionMcpServer::Impl::call_api_process(const std::string &request_content) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();

    rapidjson::Document doc;
    doc.Parse(request_content.c_str());
    if (doc.HasParseError()) {
        writer.Key("status");
        writer.Int(-1);
        writer.Key("msg");
        writer.String("input request content can not be parsed into json string");
        writer.Key("mcp_server_response");
        writer.String("");
        return buf.GetString();
    }

    if (!doc.HasMember("api_url")) {
        writer.Key("status");
        writer.Int(-1);
        writer.Key("msg");
        writer.String("input request missing \"api_url\" field");
        writer.Key("mcp_server_response");
        writer.String("");
        return buf.GetString();
    }
    std::string api_url = doc["api_url"].GetString();

    if (!doc.HasMember("api_params")) {
        writer.Key("status");
        writer.Int(-1);
        writer.Key("msg");
        writer.String("input request missing \"api_params\" field");
        writer.Key("mcp_server_response");
        writer.String("");
        return buf.GetString();
    }
    std::string api_params = doc["api_params"].GetString();

    WFFacilities::WaitGroup wait_group(1);
    std::string api_tool_resp;
    auto* http_task = WFTaskFactory::create_http_task(api_url, 5, 5, [&](WFHttpTask* h_task) -> void {
        protocol::HttpResponse* resp = h_task->get_resp();
        int state = h_task->get_state();
        int error = h_task->get_error();
        auto status_code = resp->get_status_code();
        if (state != WFT_STATE_SUCCESS) {
            writer.Key("status");
            writer.Int(-1);
            writer.Key("msg");
            writer.String("mcp server response failed");
            writer.Key("mcp_server_response");
            writer.String("workflow state error");
            api_tool_resp = buf.GetString();
            return;
        }

        if (std::strcmp(status_code, "200") != 0) {
            writer.Key("status");
            writer.Int(-1);
            writer.Key("msg");
            writer.String("mcp server response failed");
            writer.Key("mcp_server_response");
            writer.String("workflow state error");
            api_tool_resp = buf.GetString();
            return;
        }
    });


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