/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: enlighten_gan_server.cpp
* Date: 22-7-04
************************************************/

#include "enlighten_gan_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"

#include "common/base64.h"
#include "common/status_code.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "server/base_server_impl.h"
#include "factory/enhancement_task.h"

namespace jinq {
namespace server {

using jinq::common::base64;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::server::BaseAiServerImpl;

namespace enhancement {

using jinq::factory::enhancement::create_enlightengan_enhancementor;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::enhancement::std_enhancement_output;
using EnlightenGanPtr = decltype(create_enlightengan_enhancementor<base64_input, std_enhancement_output>(""));

/************ Impl Declaration ************/

class EnlightenGanServer::Impl : public BaseAiServerImpl<EnlightenGanPtr, std_enhancement_output> {
public:
    /***
    *
    * @param cfg_file_path
    * @return
    */
    StatusCode init(const toml::table& config) override;

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
StatusCode EnlightenGanServer::Impl::init(const toml::table &config) {
    // init working queue
    const toml::table* server_section_ptr = config["ENLIGHTEN_GAN_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section ENLIGHTEN_GAN_SERVER missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    const toml::table& server_section = *server_section_ptr;

    auto common_status = parse_common_server_config(server_section);
    if (common_status != StatusCode::OK) {
        return common_status;
    }
    auto worker_nums = parse_worker_nums(server_section);
        if (worker_nums <= 0) {
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg_path = config["ENLIGHTEN_GAN"]["model_config_file_path"].value_or<std::string>("");

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << "enlighten gan model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg_parsed = toml::parse_file(model_cfg_path);
    if (!model_cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(model_cfg_parsed.error().description());
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg = std::move(model_cfg_parsed).table();

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_enlightengan_enhancementor<base64_input, std_enhancement_output>(
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
        _m_model_run_timeout = static_cast<int>(server_section["model_run_timeout"].value_or<int64_t>(0));
    }

    // init server uri
    if (!server_section.contains("server_uri")) {
        LOG(ERROR) << "missing server uri field";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    } else {
        _m_server_uri = server_section["server_uri"].value_or<std::string>("");
    }

    // commit the worker watermark only after the queue is fully filled
    _m_worker_nums = static_cast<size_t>(worker_nums);
    _m_successfully_initialized = true;
    LOG(INFO) << "enlighten gan server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string EnlightenGanServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_enhancement_output& model_output) {
    int code = jinq::common::to_underlying(status);
    std::string msg = status == StatusCode::OK ? "success" : jinq::common::status_code_to_str(status);

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
    if (model_output.enhancement_result.empty()) {
        writer.String("");
    } else {
        std::vector<uchar> imencode_buffer;
        cv::imencode(".jpg", model_output.enhancement_result, imencode_buffer);
        auto output_image_data = base64::encode(imencode_buffer.data(), imencode_buffer.size());
        writer.String(output_image_data.c_str());
    }
    writer.EndObject();
    writer.EndObject();

    return buf.GetString();
}

/***
 *
 */
EnlightenGanServer::EnlightenGanServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
EnlightenGanServer::~EnlightenGanServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode EnlightenGanServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init enlighten gan derain server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
}

/***
 *
 * @param task
 */
void EnlightenGanServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool EnlightenGanServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}