/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * File: generic_ai_server.h
 * Date: 2026-08-19
 *
 * Registry-driven generic model server: the single implementation that all
 * 22 former hand-written concrete servers delegate to. Per-model variation
 * lives in AiServerSpec (TOML sections, worker factory, response filler).
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_GENERIC_AI_SERVER_H
#define MORTRED_MODEL_SERVER_GENERIC_AI_SERVER_H

#include <functional>
#include <memory>
#include <string>

#include "toml/toml.hpp"
#include "rapidjson/document.h"
#include "workflow/WFHttpServer.h"

#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "server/abstract_server.h"
#include "server/base_server_impl.h"

namespace jinq {
namespace server {

using Base64Input = jinq::models::io_define::common_io::base64_input;

template<typename MODEL_OUTPUT>
using AiWorkerPtr = std::unique_ptr<jinq::models::BaseAiModel<Base64Input, MODEL_OUTPUT>>;

template<typename MODEL_OUTPUT>
using AiWorkerFactory = std::function<AiWorkerPtr<MODEL_OUTPUT>(const std::string&)>;

template<typename MODEL_OUTPUT>
using AiResponseFiller = void (*)(rapidjson::Document::AllocatorType&,
                                  rapidjson::Document&,
                                  const MODEL_OUTPUT&);

/***
 * Per-model server registration entry. This is the whole per-model footprint
 * that used to be a ~200-line Impl copy.
 */
template<typename MODEL_OUTPUT>
struct AiServerSpec {
    std::string server_section;        // e.g. "YOLOV8_DETECTION_SERVER"
    std::string model_section;         // e.g. "YOLOV8" (holds model_config_file_path)
    std::string display_name;          // e.g. "Yolov8 object detection"
    AiWorkerFactory<MODEL_OUTPUT> make_worker;
    AiResponseFiller<MODEL_OUTPUT> fill_response;
};

template<typename MODEL_OUTPUT>
class AiModelServer final : public BaseAiServer {
  public:
    explicit AiModelServer(AiServerSpec<MODEL_OUTPUT> spec)
        : _m_spec(std::move(spec)), _m_impl(std::make_unique<Impl>(_m_spec)) {}

    AiModelServer(const AiModelServer&) = delete;
    AiModelServer& operator=(const AiModelServer&) = delete;

    jinq::common::StatusCode init(const toml::table& config) override {
        auto status = _m_impl->init(config);
        if (status != jinq::common::StatusCode::OK) {
            LOG(INFO) << "init " << _m_spec.display_name << " server failed";
            return status;
        }
        return init_http_server(_m_impl.get());
    }

    void serve_process(WFHttpTask* task) override {
        _m_impl->serve_process(task);
    }

    bool is_successfully_initialized() const override {
        return _m_impl->is_successfully_initialized();
    }

  private:
    class Impl : public BaseAiServerImpl<AiWorkerPtr<MODEL_OUTPUT>, MODEL_OUTPUT> {
      public:
        explicit Impl(const AiServerSpec<MODEL_OUTPUT>& spec) : _m_spec(spec) {}

        jinq::common::StatusCode init(const toml::table& config) override;

        void fill_response_data(rapidjson::Document::AllocatorType& allocator,
                                rapidjson::Document& data,
                                const jinq::common::StatusCode& status,
                                const MODEL_OUTPUT& model_output) override {
            (void)status;  // 契约：仅成功路径调用
            _m_spec.fill_response(allocator, data, model_output);
        }

      private:
        const AiServerSpec<MODEL_OUTPUT>& _m_spec;
    };

    AiServerSpec<MODEL_OUTPUT> _m_spec;
    std::unique_ptr<Impl> _m_impl;
};

/*********** Public Func Sets **************/

template<typename MODEL_OUTPUT>
jinq::common::StatusCode AiModelServer<MODEL_OUTPUT>::Impl::init(const toml::table& config) {
    const toml::table* server_section_ptr = config[_m_spec.server_section].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section " << _m_spec.server_section << " missing or not a table";
        this->_m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    const toml::table& server_section = *server_section_ptr;

    auto common_status = this->parse_common_server_config(server_section);
    if (common_status != jinq::common::StatusCode::OK) {
        return common_status;
    }
    auto worker_nums = parse_worker_nums(server_section);
    if (worker_nums <= 0) {
        this->_m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }

    const toml::table* model_section_ptr = config[_m_spec.model_section].as_table();
    if (model_section_ptr == nullptr) {
        LOG(ERROR) << "Config section " << _m_spec.model_section << " missing or not a table";
        this->_m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg_path = (*model_section_ptr)["model_config_file_path"].value_or<std::string>("");
    if (!jinq::common::FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << _m_spec.display_name << " model config file not exist: " << model_cfg_path;
        this->_m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg_parsed = toml::parse_file(model_cfg_path);
    if (!model_cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(model_cfg_parsed.error().description());
        this->_m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg = std::move(model_cfg_parsed).table();

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = _m_spec.make_worker("worker_" + std::to_string(index + 1));
        if (!worker->is_successfully_initialized()) {
            if (worker->init(model_cfg) != jinq::common::StatusCode::OK) {
                this->_m_successfully_initialized = false;
                return jinq::common::StatusCode::SERVER_INIT_FAILED;
            }
        }
        this->_m_working_queue.enqueue(std::move(worker));
    }

    // init server uri
    if (!server_section.contains("server_uri")) {
        LOG(ERROR) << "missing server uri field";
        this->_m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    this->_m_server_uri = server_section["server_uri"].value_or<std::string>("");

    // commit the worker watermark only after the queue is fully filled
    this->_m_worker_nums = static_cast<size_t>(worker_nums);
    this->_m_successfully_initialized = true;
    LOG(INFO) << _m_spec.display_name << " server init successfully";
    return jinq::common::StatusCode::OK;
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_MODEL_SERVER_GENERIC_AI_SERVER_H
