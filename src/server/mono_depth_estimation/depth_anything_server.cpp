/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: DepthAnythingServer.cpp
 * Date: 24-1-26
 ************************************************/

#include "depth_anything_server.h"

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
#include "factory/mono_depth_estimate_task.h"

namespace jinq {
namespace server {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::base64;
using jinq::server::BaseAiServerImpl;

namespace mono_depth_estimation {

using jinq::factory::mono_depth_estimation::create_depth_anything_estimator;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::mono_depth_estimation::std_mde_output;
using DepthAnythingPtr = decltype(create_depth_anything_estimator<base64_input, std_mde_output>(""));

/************ Impl Declaration ************/

class DepthAnythingServer::Impl : public BaseAiServerImpl<DepthAnythingPtr, std_mde_output> {
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
        const std_mde_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode DepthAnythingServer::Impl::init(const toml::table &config) {
    // init working queue
    const toml::table* server_section_ptr = config["DEPTH_ANYTHING_ESTIMATION_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section DEPTH_ANYTHING_ESTIMATION_SERVER missing or not a table";
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
    auto model_section = config["DEPTH_ANYTHING"];
    auto model_cfg_path = model_section["model_config_file_path"].value_or<std::string>("");

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << "depth anything model config file not exist: " << model_cfg_path;
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
        auto worker = create_depth_anything_estimator<base64_input, std_mde_output >(
            "worker_" + std::to_string(index + 1));

        if (!worker->is_successfully_initialized()) {
            if (worker->init(model_cfg) != StatusCode::OK) {
                _m_successfully_initialized = false;
                return StatusCode::SERVER_INIT_FAILED;
            }
        }

        _m_working_queue.enqueue(std::move(worker));
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
    LOG(INFO) << "depth anything estimation server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string DepthAnythingServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_mde_output& model_output) {
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
    // write bbox
    // write down data
    writer.Key("data");
    writer.StartObject();
    writer.Key("estimate_result");
    if (model_output.colorized_depth_map.empty()) {
        writer.String("");
    } else {
        std::vector<uchar> imencode_buffer;
        cv::imencode(".png", model_output.colorized_depth_map, imencode_buffer);
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
DepthAnythingServer::DepthAnythingServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
DepthAnythingServer::~DepthAnythingServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode DepthAnythingServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init depth anything estimation server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
}

/***
 *
 * @param task
 */
void DepthAnythingServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool DepthAnythingServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}