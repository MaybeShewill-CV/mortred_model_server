/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: pphuman_seg_server.cpp
* Date: 22-7-22
************************************************/

#include "pphuman_seg_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"

#include "common/base64.h"
#include "common/status_code.h"
#include "common/file_path_util.h"
#include "common/cv_utils.h"
#include "models/model_io_define.h"
#include "server/base_server_impl.h"
#include "factory/scene_segmentation_task.h"

namespace jinq {
namespace server {

using jinq::common::Base64;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::CvUtils;
using jinq::server::BaseAiServerImpl;

namespace scene_segmentation {

using jinq::factory::scene_segmentation::create_pphuman_segmentor;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
using PPHumanSegPtr = decltype(create_pphuman_segmentor<base64_input, std_scene_segmentation_output>(""));

/************ Impl Declaration ************/

class PPHumanSegServer::Impl : public BaseAiServerImpl<PPHumanSegPtr, std_scene_segmentation_output> {
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
        const std_scene_segmentation_output& model_output) override;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode PPHumanSegServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto server_section = config.at("PPHUMAN_SEG_SERVER");
    auto worker_nums = static_cast<int>(server_section.at("worker_nums").as_integer());
    auto model_cfg_path = config.at("PPHUMAN_SEG").at("model_config_file_path").as_string();

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(FATAL) << "pphuman seg model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg = toml::parse(model_cfg_path);

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_pphuman_segmentor<base64_input, std_scene_segmentation_output>(
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
    if (!server_section.contains("server_uri")) {
        LOG(ERROR) << "missing server uri field";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    } else {
        _m_server_uri = server_section.at("server_uri").as_string();
    }

    // init server params
    max_connection_nums = static_cast<int>(server_section.at("max_connections").as_integer());
    peer_resp_timeout = static_cast<int>(server_section.at("peer_resp_timeout").as_integer()) * 1000;
    compute_threads = static_cast<int>(server_section.at("compute_threads").as_integer());
    handler_threads = static_cast<int>(server_section.at("handler_threads").as_integer());
    request_size_limit = static_cast<size_t>(server_section.at("request_size_limit").as_integer());

    _m_successfully_initialized = true;
    LOG(INFO) << "pphuman segmentation server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string PPHumanSegServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_scene_segmentation_output& model_output) {
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
    writer.Key("segment_result");
    if (model_output.segmentation_result.empty()) {
        writer.String("");
    } else {
        std::vector<uchar> imencode_buffer;
        cv::imencode(".png", model_output.segmentation_result, imencode_buffer);
        auto output_image_data = Base64::base64_encode(imencode_buffer.data(), imencode_buffer.size());
        writer.String(output_image_data.c_str());
    }
    writer.Key("colorized_seg_mask");
    if (model_output.segmentation_result.empty()) {
        writer.String("");
    } else {
        cv::Mat color_mask;
        CvUtils::colorize_segmentation_mask(model_output.segmentation_result, color_mask, 80);
        std::vector<uchar> imencode_buffer;
        cv::imencode(".png", color_mask, imencode_buffer);
        auto output_image_data = Base64::base64_encode(imencode_buffer.data(), imencode_buffer.size());
        writer.String(output_image_data.c_str());
    }

    writer.EndObject();
    writer.EndObject();

    return buf.GetString();
}

/***
 *
 */
PPHumanSegServer::PPHumanSegServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
PPHumanSegServer::~PPHumanSegServer() = default;

/***
 *
 * @param cfg
 * @return
 */
jinq::common::StatusCode PPHumanSegServer::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);

    if (status != StatusCode::OK) {
        LOG(INFO) << "init pphuman segmentation server failed";
        return status;
    }

    // init server
    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = _m_impl->compute_threads;
    settings.handler_threads = _m_impl->handler_threads;
    WORKFLOW_library_init(&settings);

    WFServerParams server_params = SERVER_PARAMS_DEFAULT;
    server_params.max_connections = _m_impl->max_connection_nums;
    server_params.peer_response_timeout = _m_impl->peer_resp_timeout;
    server_params.request_size_limit = _m_impl->request_size_limit * 1024 * 1024;

    auto&& proc = [&](auto arg) { return this->_m_impl->serve_process(arg); };
    _m_server = std::make_unique<WFHttpServer>(proc);

    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void PPHumanSegServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool PPHumanSegServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}