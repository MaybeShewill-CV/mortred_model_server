/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: resnet_server.cpp
* Date: 22-6-21
************************************************/

#include "resnet_server.h"

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
#include "factory/classification_task.h"

namespace jinq {
namespace server {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::server::BaseAiServerImpl;

namespace classification {

using jinq::factory::classification::create_resnet_classifier;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::classification::std_classification_output;
using ResNetPtr = decltype(create_resnet_classifier<base64_input, std_classification_output>(""));

/************ Impl Declaration ************/

class ResNetServer::Impl : public BaseAiServerImpl<ResNetPtr, std_classification_output> {
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
        const std_classification_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode ResNetServer::Impl::init(const toml::table &config) {
    // init working queue
    const toml::table* server_section_ptr = config["RESNET_CLASSIFICATION_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section RESNET_CLASSIFICATION_SERVER missing or not a table";
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
    auto model_section = config["RESNET"];
    auto model_cfg_path = model_section["model_config_file_path"].value_or<std::string>("");

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << "resnet model config file not exist: " << model_cfg_path;
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
        auto worker = create_resnet_classifier<base64_input, std_classification_output>(
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

    _m_successfully_initialized = true;
    LOG(INFO) << "Resnet classification server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string ResNetServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_classification_output& model_output) {
    int code = jinq::common::to_underlying(status);
    std::string msg = status == StatusCode::OK ? "success" : jinq::common::status_code_to_str(status);
    int cls_id = -1;
    float scores = -1.0;
    std::string category;

    if (status == StatusCode::OK) {
        cls_id = model_output.class_id;
        if (cls_id >= 0 && cls_id < static_cast<int>(model_output.scores.size())) {
            scores = model_output.scores[cls_id];
        }
        category = model_output.category;
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
    writer.Key("category");
    writer.String(category.c_str());
    writer.EndObject();
    writer.EndObject();

    return buf.GetString();
}

/****************** ResNetServer Implementation **************/

/***
 *
 */
ResNetServer::ResNetServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
ResNetServer::~ResNetServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode ResNetServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);
    if (status != StatusCode::OK) {
        LOG(INFO) << "init resnet classification server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
}

/***
 *
 * @param task
 */
void ResNetServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool ResNetServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}
