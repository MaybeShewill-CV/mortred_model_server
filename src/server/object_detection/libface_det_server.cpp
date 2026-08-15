/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: libface_det_server.cpp
* Date: 22-6-26
************************************************/

#include "libface_det_server.h"

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
#include "factory/obj_detection_task.h"

namespace jinq {
namespace server {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::server::BaseAiServerImpl;

namespace object_detection {

using jinq::factory::object_detection::create_libface_detector;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::object_detection::std_face_detection_output;
using LibfaceDetPtr = decltype(create_libface_detector<base64_input, std_face_detection_output>(""));

/************ Impl Declaration ************/

class LibfaceDetServer::Impl : public BaseAiServerImpl<LibfaceDetPtr, std_face_detection_output> {
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
        const std_face_detection_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode LibfaceDetServer::Impl::init(const toml::table &config) {
    // init working queue
    const toml::table* server_section_ptr = config["LIBFACE_DETECTION_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section LIBFACE_DETECTION_SERVER missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    const toml::table& server_section = *server_section_ptr;

    auto common_status = parse_common_server_config(server_section);
    if (common_status != StatusCode::OK) {
        return common_status;
    }
    auto worker_nums = parse_worker_nums(server_section);
    _m_worker_nums = worker_nums > 0 ? static_cast<size_t>(worker_nums) : 0;
    if (worker_nums <= 0) {
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg_path = config["LIBFACE"]["model_config_file_path"].value_or<std::string>("");

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << "LIBFACE model config file not exist: " << model_cfg_path;
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
        auto worker = create_libface_detector<base64_input, std_face_detection_output>(
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
    LOG(INFO) << "libface object detection server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string LibfaceDetServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_face_detection_output& model_output) {
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
    writer.StartArray();

    for (auto& obj_box : model_output) {
        int cls_id = obj_box.class_id;
        auto score = obj_box.score;
        auto category = obj_box.category;

        writer.StartObject();
        // write obj cls id
        writer.Key("cls_id");
        writer.String(std::to_string(cls_id).c_str());
        // write obj score
        writer.Key("score");
        writer.String(std::to_string(score).c_str());
        // write obj category
        writer.Key("category");
        writer.String(category.c_str());
        // write obj point coords
        writer.Key("box");
        writer.StartArray();
        // left top coords
        writer.StartArray();
        writer.Double(obj_box.bbox.x);
        writer.Double(obj_box.bbox.y);
        writer.EndArray();
        // right bottom coords
        writer.StartArray();
        writer.Double(obj_box.bbox.x + obj_box.bbox.width);
        writer.Double(obj_box.bbox.y + obj_box.bbox.height);
        writer.EndArray();
        writer.EndArray();
        // write face landmarks
        writer.Key("landmark");
        writer.StartArray();

        for (const auto& pt : obj_box.landmarks) {
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
LibfaceDetServer::LibfaceDetServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
LibfaceDetServer::~LibfaceDetServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode LibfaceDetServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init libface detection server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
}

/***
 *
 * @param task
 */
void LibfaceDetServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool LibfaceDetServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}