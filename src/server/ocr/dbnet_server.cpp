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

using jinq::common::base64;
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
        const std_text_regions_output & model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode DBNetServer::Impl::init(const toml::table &config) {
    // init working queue
    const toml::table* server_section_ptr = config["DBNET_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section DBNET_SERVER missing or not a table";
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
    auto model_cfg_path = config["DBNET"]["model_config_file_path"].value_or<std::string>("");

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << "dbnet model config file not exist: " << model_cfg_path;
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
jinq::common::StatusCode DBNetServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init dbnet server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
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