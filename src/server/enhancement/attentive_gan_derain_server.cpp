/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: attentive_gan_derain_server.cpp
* Date: 22-7-04
************************************************/

#include "attentive_gan_derain_server.h"

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
#include "factory/enhancement_task.h"

namespace jinq {
namespace server {

using jinq::common::Base64;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::server::BaseAiServerImpl;

namespace enhancement {

using jinq::factory::enhancement::create_attentivegan_enhancementor;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::enhancement::std_enhancement_output;
using AttentiveGanPtr = decltype(create_attentivegan_enhancementor<base64_input, std_enhancement_output>(""));

/************ Impl Declaration ************/

class AttentiveGanDerainServer::Impl : public BaseAiServerImpl<AttentiveGanPtr, std_enhancement_output> {
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
        const std_enhancement_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode AttentiveGanDerainServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto server_section = config.at("ATTENTIVE_GAN_DERAIN_SERVER");
    auto worker_nums = static_cast<int>(server_section.at("worker_nums").as_integer());
    auto model_cfg_path = config.at("ATTENTIVE_GAN_DERAIN").at("model_config_file_path").as_string();

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(FATAL) << "attentive gan model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg = toml::parse(model_cfg_path);

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_attentivegan_enhancementor<base64_input, std_enhancement_output>(
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
    LOG(INFO) << "attentive gan derain server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string AttentiveGanDerainServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_enhancement_output& model_output) {
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
    writer.StartObject();
    writer.Key("enhance_result");
    std::vector<uchar> imencode_buffer;
    cv::imencode(".jpg", model_output.enhancement_result, imencode_buffer);
    auto output_image_data = Base64::base64_encode(imencode_buffer.data(), imencode_buffer.size());
    writer.String(output_image_data.c_str());
    writer.EndObject();
    writer.EndObject();

    return buf.GetString();
}

/***
 *
 */
AttentiveGanDerainServer::AttentiveGanDerainServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
AttentiveGanDerainServer::~AttentiveGanDerainServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode AttentiveGanDerainServer::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init attentive gan derain server failed";
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
            &AttentiveGanDerainServer::Impl::serve_process, std::cref(this->_m_impl), std::placeholders::_1);
    _m_server = std::make_unique<WFHttpServer>( proc);

    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void AttentiveGanDerainServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool AttentiveGanDerainServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}