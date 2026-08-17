/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: YoloV8DetServer.cpp
 * Date: 24-3-14
 ************************************************/

#include "yolov8_det_server.h"

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

using jinq::factory::object_detection::create_yolov8_detector;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::object_detection::std_object_detection_output;
using Yolov8DetPtr = decltype(create_yolov8_detector<base64_input, std_object_detection_output>(""));

class YoloV8DetServer::Impl : public BaseAiServerImpl<Yolov8DetPtr, std_object_detection_output> {
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
        const std_object_detection_output& model_output) override;
};

/***
 *
 * @param config
 * @return
 */
StatusCode YoloV8DetServer::Impl::init(const toml::table &config) {
    // init working queue
    const toml::table* server_section_ptr = config["YOLOV8_DETECTION_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section YOLOV8_DETECTION_SERVER missing or not a table";
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
    auto model_cfg_path = config["YOLOV8"]["model_config_file_path"].value_or<std::string>("");

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << "yolov8 model config file not exist: " << model_cfg_path;
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
        auto worker = create_yolov8_detector<base64_input, std_object_detection_output>(
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
    LOG(INFO) << "Yolov8 object detection server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
void YoloV8DetServer::Impl::fill_response_data(
    rapidjson::Document::AllocatorType& allocator,
    rapidjson::Document& data,
    const StatusCode& status,
    const std_object_detection_output& model_output) {
    data.SetArray();
    if (status != StatusCode::OK) {
        return;
    }
    for (const auto& obj_box : model_output) {
        rapidjson::Value item(rapidjson::kObjectType);
        item.AddMember("cls_id", obj_box.class_id, allocator);
        item.AddMember("score", obj_box.score, allocator);
        item.AddMember("category",
                       rapidjson::Value(obj_box.category.c_str(),
                                        obj_box.category.size(),
                                        allocator),
                       allocator);

        rapidjson::Value points(rapidjson::kArrayType);
        rapidjson::Value left_top(rapidjson::kArrayType);
        left_top.PushBack(obj_box.bbox.x, allocator);
        left_top.PushBack(obj_box.bbox.y, allocator);
        rapidjson::Value right_bottom(rapidjson::kArrayType);
        right_bottom.PushBack(obj_box.bbox.x + obj_box.bbox.width, allocator);
        right_bottom.PushBack(obj_box.bbox.y + obj_box.bbox.height, allocator);
        points.PushBack(left_top, allocator);
        points.PushBack(right_bottom, allocator);
        item.AddMember("points", points, allocator);

        item.AddMember("detail_infos", rapidjson::Value(rapidjson::kObjectType), allocator);
        data.PushBack(item, allocator);
    }
}

/***
 *
 */
YoloV8DetServer::YoloV8DetServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
YoloV8DetServer::~YoloV8DetServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode YoloV8DetServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);
    if (status != StatusCode::OK) {
        LOG(INFO) << "init yolov8 detection server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
}

/***
 *
 * @param task
 */
void YoloV8DetServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool YoloV8DetServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}

}
}
}
