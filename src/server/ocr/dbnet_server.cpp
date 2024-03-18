/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dbnet_server.cpp
* Date: 22-7-04
************************************************/

#include "dbnet_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "workflow/Workflow.h"

#include "common/base64.h"
#include "common/status_code.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "server/base_server_impl.h"
#include "factory/ocr_task.h"

namespace jinq {
namespace server {

using jinq::common::Base64;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::server::BaseAiServerImpl;

namespace ocr {

using jinq::factory::ocr::create_dbtext_detector;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::ocr::std_text_regions_output;
using DBNetPtr = decltype(create_dbtext_detector<base64_input, std_text_regions_output>(""));

/************ Impl Declaration ************/

class DBNetServer::Impl : public BaseAiServerImpl<DBNetPtr, std_text_regions_output> {
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
        const std_text_regions_output & model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode DBNetServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto server_section = config.at("DBNET_SERVER");
    auto worker_nums = static_cast<int>(server_section.at("worker_nums").as_integer());
    auto model_cfg_path = config.at("DBNET").at("model_config_file_path").as_string();

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(FATAL) << "dbnet model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg = toml::parse(model_cfg_path);

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_dbtext_detector<base64_input, std_text_regions_output>(
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
    request_size_limit = static_cast<size_t>(server_section.at("request_size_limit").as_integer());

    _m_successfully_initialized = true;
    LOG(INFO) << "dbnet server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string DBNetServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_text_regions_output & model_output) {
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
    // write output
    writer.Key("data");
    writer.StartArray();
    for (const auto& region : model_output) {
        auto score = region.score;
        auto bbox = region.bbox;
        auto polygon = region.polygon;
        // write score
        writer.StartObject();
        writer.Key("score");
        writer.Double(score);
        // write bbox
        writer.Key("bbox");
        writer.StartArray();
        // left top coords
        writer.StartArray();
        writer.Double(bbox.x);
        writer.Double(bbox.y);
        writer.EndArray();
        // right bottom coords
        writer.StartArray();
        writer.Double(bbox.x + bbox.width);
        writer.Double(bbox.y + bbox.height);
        writer.EndArray();
        writer.EndArray();
        // write text region polygon
        writer.Key("polygon");
        writer.StartArray();
        for (const auto& pt : polygon) {
            writer.StartArray();
            writer.Double(pt.x);
            writer.Double(pt.y);
            writer.EndArray();
        }
        writer.EndArray();
        // write extra detail infos
        writer.Key("detail_infos");
        writer.StartObject();
        writer.EndObject();

        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();

    return buf.GetString();
}

/***
 *
 */
DBNetServer::DBNetServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
DBNetServer::~DBNetServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode DBNetServer::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init dbnet server failed";
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

    auto&& proc = [&](auto arg) { return this->_m_impl->serve_process(arg); };
    _m_server = std::make_unique<WFHttpServer>(&server_params, proc);

    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void DBNetServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool DBNetServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}