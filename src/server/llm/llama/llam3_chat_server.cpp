/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llam3_chat_server.cpp
 * Date: 24-11-29
 ************************************************/

#include "llam3_chat_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "workflow/Workflow.h"

#include "common/status_code.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "models/llm/llama/llama3_generator.h"
#include "server/base_server_impl.h"

namespace jinq {
namespace server {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::server::BaseAiServerImpl;

namespace llm {

using models::llm::llama::

namespace llama3 {

/************ Impl Declaration ************/

class Llama3ChatServer::Impl {
public:
    /***
    *
    * @param cfg_file_path
    * @return
     */
    StatusCode init(const decltype(toml::parse("")) &config);

private:
    /***
     *
     * @param task_id
     * @param status
     * @param gen_output
     * @return
     */
    std::string make_response_body(const std::string& task_id, const StatusCode& status, const std::string& gen_output);

  private:
    // init flag
    bool _m_successfully_initialized = false;
    // task count
    std::atomic<size_t> _m_received_jobs{0};
    std::atomic<size_t> _m_finished_jobs{0};
    std::atomic<size_t> _m_waiting_jobs{0};
    // model run timeout
    int _m_model_run_timeout = 500; // ms
    // server uri
    std::string _m_server_uri;
    // llama3 generator
    Lla

    // server params
    int _m_max_connection_nums = 200;
    int _m_peer_resp_timeout = 15 * 1000;
    int _m_compute_threads = -1;
    int _m_handler_threads = 50;
    size_t _m_request_size_limit = -1;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode Llama3ChatServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto server_section = config.at("LLAMA3_CHAT_SERVER");
    auto worker_nums = static_cast<int>(server_section.at("worker_nums").as_integer());
    auto model_section = config.at("LLAMA3_CHAT_MODEL");
    auto model_cfg_path = model_section.at("model_config_file_path").as_string();

    // init server params
    _m_max_connection_nums = static_cast<int>(server_section.at("max_connections").as_integer());
    _m_peer_resp_timeout = static_cast<int>(server_section.at("peer_resp_timeout").as_integer()) * 1000;
    _m_compute_threads = static_cast<int>(server_section.at("compute_threads").as_integer());
    _m_handler_threads = static_cast<int>(server_section.at("handler_threads").as_integer());
    _m_request_size_limit = static_cast<size_t>(server_section.at("request_size_limit").as_integer());

    _m_successfully_initialized = true;
    LOG(INFO) << "Superpoint feature point detection server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string Llama3ChatServer::Impl::make_response_body(const std::string& task_id, const StatusCode& status,
        const std_feature_point_output& model_output) {
    int code = static_cast<int>(status);
    std::string msg = status == StatusCode::OK ? "success" : jinq::common::error_code_to_str(code);

    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();
    // write req id
    writer.Key("req_id");
    writer.String(task_id.c_str());
    // write code
    writer.Key("code");
    writer.Int(code);
    // write msg
    writer.Key("msg");
    writer.String(msg.c_str());
    // write bbox
    // write down data
    writer.Key("data");
    writer.StartArray();

    for (auto& fp : model_output) {
        // fille in fp conf score
        writer.Key("score");
        writer.Double(fp.score);
        // fill in fp localtion
        writer.Key("location");
        writer.StartArray();
        writer.Double(fp.location.x);
        writer.Double(fp.location.y);
        writer.EndArray();
        // fille in fp descriptor
        writer.Key("descriptor");
        writer.StartArray();
        // for (const auto& ft_val : fp.descriptor) {
        //     writer.Double(ft_val);
        // }
        writer.EndArray();
    }

    writer.EndArray();
    writer.EndObject();

    return buf.GetString();
}

/***
 *
 */
Llama3ChatServer::Llama3ChatServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
Llama3ChatServer::~Llama3ChatServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode Llama3ChatServer::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init superpoint detection server failed";
        return status;
    }

    // init server
    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = _m_impl->compute_threads;
    settings.handler_threads = _m_impl->handler_threads;
    WORKFLOW_library_init(&settings);

    WFServerParams server_params = SERVER_PARAMS_DEFAULT;
    server_params.max_connections = _m_impl->max_connection_nums;
    server_params.peer_response_timeout = _m_impl->peer_resp_timeout;
    server_params.request_size_limit = _m_impl->request_size_limit * 1024 * 1024;

    auto&& proc = [&](auto arg) {
        return this->_m_impl->serve_process(arg);
    };
    _m_server = std::make_unique<WFHttpServer>(&server_params, proc);

    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void Llama3ChatServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool Llama3ChatServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}

}
}
}
}
