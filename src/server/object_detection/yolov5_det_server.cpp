/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: Yolov5DetServer.cpp
* Date: 22-6-22
************************************************/

#include "yolov5_det_server.h"

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
#include "factory/obj_detection_task.h"

namespace mortred {
namespace server {

using mortred::common::FilePathUtil;
using mortred::common::StatusCode;
using mortred::server::BaseAiServerImpl;

namespace object_detection {

using mortred::factory::object_detection::create_yolov5_detector;
using mortred::models::io_define::common_io::base64_input;
using mortred::models::io_define::object_detection::std_object_detection_output;
using Yolov5DetPtr = decltype(create_yolov5_detector<base64_input, std_object_detection_output>(""));

class YoloV5DetServer::Impl : public BaseAiServerImpl<Yolov5DetPtr, std_object_detection_output> {
public:
    /***
    *
    * @param cfg_file_path
    * @return
    */
    StatusCode init(const decltype(toml::parse(""))& config);

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
        const std_object_detection_output& model_output) override;
};

/***
 *
 * @param config
 * @return
 */
StatusCode YoloV5DetServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto server_section = config.at("YOLOV5_DETECTION_SERVER");
    auto worker_nums = static_cast<int>(server_section.at("worker_nums").as_integer());
    auto model_cfg_path = config.at("YOLOV5").at("model_config_file_path").as_string();

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(FATAL) << "yolov5 model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg = toml::parse(model_cfg_path);

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_yolov5_detector<base64_input, std_object_detection_output>(
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
    LOG(INFO) << "Yolov5 object detection server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string YoloV5DetServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_object_detection_output& model_output) {
    int code = static_cast<int>(status);
    std::string msg = status == StatusCode::OK ? "success" : mortred::common::error_code_to_str(code);

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

        writer.StartObject();
        // write obj cls id
        writer.Key("cls_id");
        writer.String(std::to_string(cls_id).c_str());
        // write obj score
        writer.Key("score");
        writer.String(std::to_string(score).c_str());
        // write obj point coords
        writer.Key("points");
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
YoloV5DetServer::YoloV5DetServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
YoloV5DetServer::~YoloV5DetServer() = default;

/***
 *
 * @param cfg
 * @return
 */
mortred::common::StatusCode YoloV5DetServer::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init yolov5 detection server failed";
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
            &YoloV5DetServer::Impl::serve_process, std::cref(this->_m_impl), std::placeholders::_1);
    _m_server = std::make_unique<WFHttpServer>(proc);

    // init _m_impl
    return _m_impl->init(config);
}

/***
 *
 * @param task
 */
void YoloV5DetServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool YoloV5DetServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}