/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: superpoint_fp_server.cpp
* Date: 22-6-29
************************************************/

#include "superpoint_fp_server.h"

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
#include "server/base_server_impl.h"
#include "factory/feature_point_task.h"

namespace mortred {
namespace server {

using mortred::common::FilePathUtil;
using mortred::common::StatusCode;
using mortred::server::BaseAiServerImpl;

namespace feature_point {

using mortred::factory::feature_point::create_superpoint_extractor;
using mortred::models::io_define::common_io::base64_input;
using mortred::models::io_define::feature_point::std_feature_point_output;
using SuperPointPtr = decltype(create_superpoint_extractor<base64_input, std_feature_point_output>(""));

/************ Impl Declaration ************/

class SuperpointFpServer::Impl : public BaseAiServerImpl<SuperPointPtr, std_feature_point_output> {
public:
    /***
    *
    * @param cfg_file_path
    * @return
    */
    StatusCode init(const decltype(toml::parse(""))& config) override;

protected:
    /***
     *
     * @param task_id
     * @param status
     * @param model_output
     * @return
     */
    std::string make_response_body(
        const std::string& task_id,
        const StatusCode& status,
        const std_feature_point_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode SuperpointFpServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto server_section = config.at("SUPERPOINT_FP_SERVER");
    auto worker_nums = static_cast<int>(server_section.at("worker_nums").as_integer());
    auto model_section = config.at("SUPERPOINT");
    auto model_cfg_path = model_section.at("model_config_file_path").as_string();

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(FATAL) << "superpoint model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg = toml::parse(model_cfg_path);

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_superpoint_extractor<base64_input, std_feature_point_output>(
                          "worker_" + std::to_string(index + 1));

        if (!worker->is_successfully_initialized()) {
            if (worker->init(model_cfg) != StatusCode::OK) {
                _m_successfully_initialized = false;
                return StatusCode::SERVER_INIT_FAILED;
            }
        }

        _m_working_queue.enqueue(std::move(worker));
    }

    // init worker run timeout
    if (!server_section.contains("model_run_timeout")) {
        _m_model_run_timeout = 500; // ms
    } else {
        _m_model_run_timeout = static_cast<int>(server_section.at("model_run_timeout").as_integer());
    }

    // init server uri
    if (!server_section.contains("server_url")) {
        LOG(ERROR) << "missing server uri field";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    } else {
        _m_server_uri = server_section.at("server_url").as_string();
    }

    // init server params
    max_connection_nums = static_cast<int>(server_section.at("max_connections").as_integer());
    peer_resp_timeout = static_cast<int>(server_section.at("peer_resp_timeout").as_integer()) * 1000;
    compute_threads = static_cast<int>(server_section.at("compute_threads").as_integer());
    handler_threads = static_cast<int>(server_section.at("handler_threads").as_integer());

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
std::string SuperpointFpServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_feature_point_output& model_output) {
    int code = static_cast<int>(status);
    std::string msg = status == StatusCode::OK ? "success" : mortred::common::error_code_to_str(code);

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

        for (const auto& ft_val : fp.descriptor) {
            writer.Double(ft_val);
        }

        writer.EndArray();
    }

    writer.EndArray();
    writer.EndObject();

    return buf.GetString();
}

/***
 *
 */
SuperpointFpServer::SuperpointFpServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
SuperpointFpServer::~SuperpointFpServer() = default;

/***
 *
 * @param cfg
 * @return
 */
mortred::common::StatusCode SuperpointFpServer::init(const decltype(toml::parse("")) &config) {
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
    settings.endpoint_params.max_connections = _m_impl->max_connection_nums;
    settings.endpoint_params.response_timeout = _m_impl->peer_resp_timeout;
    WORKFLOW_library_init(&settings);

    auto&& proc = std::bind(
            &SuperpointFpServer::Impl::serve_process, std::cref(this->_m_impl), std::placeholders::_1);
    _m_server = std::make_unique<WFHttpServer>(proc);

    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void SuperpointFpServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool SuperpointFpServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}