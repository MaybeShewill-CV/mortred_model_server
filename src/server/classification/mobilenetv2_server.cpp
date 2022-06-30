/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2_server.inl
* Date: 22-6-21
************************************************/

#include "mobilenetv2_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "workflow/Workflow.h"

#include "common/md5.h"
#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "server/base_server_impl.h"
#include "factory/classification_task.h"

namespace mortred {
namespace server {

using mortred::common::Base64;
using mortred::common::CvUtils;
using mortred::common::FilePathUtil;
using mortred::common::Md5;
using mortred::common::StatusCode;
using mortred::common::Timestamp;
using mortred::server::BaseAiServerImpl;

namespace classification {

using mortred::factory::classification::create_mobilenetv2_classifier;
using mortred::models::io_define::common_io::base64_input;
using mortred::models::io_define::classification::std_classification_output;
using MoblieNetv2NetPtr = decltype(create_mobilenetv2_classifier<base64_input, std_classification_output>(""));

/************ Impl Declaration ************/

class MobileNetv2Server::Impl : public BaseAiServerImpl<MoblieNetv2NetPtr, std_classification_output> {
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
        const std_classification_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode MobileNetv2Server::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto worker_nums = static_cast<int>(config.at("MOBILENETV2_CLASSIFICATION_SERVER").at("worker_nums").as_integer());
    auto model_cfg_path = config.at("MOBILENETV2").at("model_config_file_path").as_string();

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(FATAL) << "mobilenetv2 model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg = toml::parse(model_cfg_path);

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_mobilenetv2_classifier<base64_input, std_classification_output>(
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
    if (!config.at("MOBILENETV2_CLASSIFICATION_SERVER").contains("model_run_timeout")) {
        _m_model_run_timeout = 500; // ms
    } else {
        _m_model_run_timeout = static_cast<int>(
                                   config.at("MOBILENETV2_CLASSIFICATION_SERVER").at("model_run_timeout").as_integer());
    }

    // init server uri
    if (!config.at("MOBILENETV2_CLASSIFICATION_SERVER").contains("server_url")) {
        LOG(ERROR) << "missing server uri field";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    } else {
        _m_server_uri = config.at("MOBILENETV2_CLASSIFICATION_SERVER").at("server_url").as_string();
    }

    // init server params
    toml::value cfg_content = config.at("MOBILENETV2_CLASSIFICATION_SERVER");
    max_connection_nums = static_cast<int>(cfg_content.at("max_connections").as_integer());
    peer_resp_timeout = static_cast<int>(cfg_content.at("peer_resp_timeout").as_integer()) * 1000;
    compute_threads = static_cast<int>(cfg_content.at("compute_threads").as_integer());
    handler_threads = static_cast<int>(cfg_content.at("handler_threads").as_integer());

    _m_successfully_initialized = true;
    LOG(INFO) << "Mobilenetv2 classification server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string MobileNetv2Server::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_classification_output& model_output) {
    int code = static_cast<int>(status);
    std::string msg = status == StatusCode::OK ? "success" : mortred::common::error_code_to_str(code);
    int cls_id = -1;
    float scores = -1.0;

    if (status == StatusCode::OK) {
        cls_id = model_output.class_id;
        scores = model_output.scores[cls_id];
    }

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
    // write class result
    writer.Key("data");
    writer.StartObject();
    writer.Key("class_id");
    writer.Int(cls_id);
    writer.Key("scores");
    writer.Double(scores);
    writer.EndObject();
    writer.EndObject();

    return buf.GetString();
}

/****************** MobileNetv2Server Implementation **************/

/***
 *
 */
MobileNetv2Server::MobileNetv2Server() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
MobileNetv2Server::~MobileNetv2Server() = default;

/***
 *
 * @param cfg
 * @return
 */
mortred::common::StatusCode MobileNetv2Server::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);
    if (status != StatusCode::OK) {
        LOG(INFO) << "init mobilenetv2 classification server failed";
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
                      &MobileNetv2Server::Impl::serve_process, std::cref(this->_m_impl), std::placeholders::_1);
    _m_server = std::make_unique<WFHttpServer>(proc);

    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void MobileNetv2Server::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool MobileNetv2Server::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}
