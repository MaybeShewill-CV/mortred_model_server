/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: densenet_server.cpp
* Date: 22-7-1
************************************************/

#include "densenet_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"

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

using jinq::factory::classification::create_densenet_classifier;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::classification::std_classification_output;
using DenseNetPtr = decltype(create_densenet_classifier<base64_input, std_classification_output>(""));

/************ Impl Declaration ************/

class DenseNetServer::Impl : public BaseAiServerImpl<DenseNetPtr, std_classification_output> {
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
    void fill_response_data(
        rapidjson::Document::AllocatorType& allocator,
        rapidjson::Document& data,
        const StatusCode& status,
        const std_classification_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode DenseNetServer::Impl::init(const toml::table &config) {
    // init working queue
    const toml::table* server_section_ptr = config["DENSENET_CLASSIFICATION_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section DENSENET_CLASSIFICATION_SERVER missing or not a table";
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
    auto model_section = config["DENSENET"];
    auto model_cfg_path = model_section["model_config_file_path"].value_or<std::string>("");

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << "densenet model config file not exist: " << model_cfg_path;
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
        auto worker = create_densenet_classifier<base64_input, std_classification_output>(
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
    LOG(INFO) << "densenet classification server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
void DenseNetServer::Impl::fill_response_data(
    rapidjson::Document::AllocatorType& allocator,
    rapidjson::Document& data,
    const StatusCode& status,
    const std_classification_output& model_output) {
    data.SetObject();
    if (status != StatusCode::OK) {
        return;
    }
    data.AddMember("class_id", model_output.class_id, allocator);
    data.AddMember("category",
                   rapidjson::Value(model_output.category.c_str(),
                                    model_output.category.size(),
                                    allocator),
                   allocator);
    rapidjson::Value scores(rapidjson::kArrayType);
    for (float s : model_output.scores) {
        scores.PushBack(s, allocator);
    }
    data.AddMember("scores", scores, allocator);
}

/***
 *
 */
DenseNetServer::DenseNetServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
DenseNetServer::~DenseNetServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode DenseNetServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);
    if (status != StatusCode::OK) {
        LOG(INFO) << "init densenet classification server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
}

/***
 *
 * @param task
 */
void DenseNetServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool DenseNetServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}
