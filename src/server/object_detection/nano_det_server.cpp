/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: nanodet_server.cpp
* Date: 22-6-21
************************************************/

#include "nano_det_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "stl_container/concurrentqueue.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "workflow/Workflow.h"

#include "common/md5.h"
#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "factory/obj_detection_task.h"

namespace mortred {
namespace server {

using mortred::common::Base64;
using mortred::common::CvUtils;
using mortred::common::FilePathUtil;
using mortred::common::Md5;
using mortred::common::StatusCode;
using mortred::common::Timestamp;

namespace object_detection {

using mortred::factory::object_detection::create_nanodet_detector;
using mortred::models::io_define::common_io::base64_input;
using mortred::models::io_define::object_detection::std_object_detection_output;
using NanoDetPtr = decltype(create_nanodet_detector<base64_input, std_object_detection_output>(""));

/************ Impl Declaration ************/

class NanoDetServer::Impl {
public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
    *
    * @param transformer
    */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
    *
    * @param cfg_file_path
    * @return
    */
    StatusCode init(const decltype(toml::parse(""))& config);

    /***
     *
     * @param task
     */
    void serve_process(WFHttpTask* task);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

private:
    // init flag
    bool _m_successfully_initialized = false;
    // task count
    std::atomic<size_t> _m_received_jobs{0};
    std::atomic<size_t> _m_finished_jobs{0};
    std::atomic<size_t> _m_waiting_jobs{0};
    // worker queue
    moodycamel::ConcurrentQueue<NanoDetPtr> _m_working_queue;
private:
    struct seriex_ctx {
        protocol::HttpResponse* response = nullptr;
    };

    struct det_request {
        std::string image_content;
        std::string task_id;
        bool is_valid = true;
    };

private:
    /***
     *
     * @param req_body
     * @return
     */
    static det_request parse_task_request(const std::string& req_body);

    /***
     *
     * @param task_id
     * @param status
     * @param model_output
     * @return
     */
    static std::string make_response_body(
        const std::string& task_id,
        const StatusCode& status,
        const std_object_detection_output& model_output);

    /***
     *
     * @param req
     * @param ctx
     */
    void do_detection(const det_request& req, seriex_ctx* ctx);
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode NanoDetServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    auto worker_nums = static_cast<int>(config.at("NANODET_DETECTION_SERVER").at("worker_nums").as_integer());
    auto model_cfg_path = config.at("NANODET").at("model_config_file_path").as_string();

    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(FATAL) << "nanodet model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg = toml::parse(model_cfg_path);

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = create_nanodet_detector<base64_input, std_object_detection_output>(
                          "worker_" + std::to_string(index + 1));
        if (!worker->is_successfully_initialized()) {
            if (worker->init(model_cfg) != StatusCode::OK) {
                _m_successfully_initialized = false;
                return StatusCode::SERVER_INIT_FAILED;
            }
        }
        _m_working_queue.enqueue(std::move(worker));
    }

    _m_successfully_initialized = true;
    LOG(INFO) << "NanoDet object detection server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void NanoDetServer::Impl::serve_process(WFHttpTask* task) {
    // welcome message
    if (strcmp(task->get_req()->get_request_uri(), "/welcome") == 0) {
        task->get_resp()->append_output_body("<html>Welcome to mortred NanoDet Object Detection Server</html>");
        return;
    }

    // hello world message
    if (strcmp(task->get_req()->get_request_uri(), "/hello_world") == 0) {
        task->get_resp()->append_output_body("<html>Hello World !!!</html>");
        return;
    }

    // nanodet obj detection
    if (strcmp(task->get_req()->get_request_uri(), "/mortred_ai_server_v1/obj_detection/nanodet") == 0) {
        // parse request body
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto det_task_req = parse_task_request(protocol::HttpUtil::decode_chunked_body(req));
        _m_waiting_jobs++;
        _m_received_jobs++;
        // init series work
        auto* series = series_of(task);
        auto* ctx = new seriex_ctx;
        ctx->response = resp;
        series->set_context(ctx);
        series->set_callback([](const SeriesWork * series) {
            delete (seriex_ctx*)series->get_context();
        });
        // do classification
        auto&& go_proc = std::bind(&NanoDetServer::Impl::do_detection, this, det_task_req, ctx);
        auto* cls_task = WFTaskFactory::create_go_task("nanodet_det_work", go_proc, det_task_req, ctx);
        *series << cls_task;
    }
}

/***
 *
 * @param req_body
 * @return
 */
NanoDetServer::Impl::det_request NanoDetServer::Impl::parse_task_request(const std::string& req_body) {
    rapidjson::Document doc;
    doc.Parse(req_body.c_str());
    det_request req{};

    if (doc.HasParseError() || doc.IsNull() || doc.ObjectEmpty()) {
        req.image_content = "";
        req.is_valid = false;
    } else {
        CHECK_EQ(doc.IsObject(), true);

        if (!doc.HasMember("img_data") || !doc["img_data"].IsString()) {
            req.image_content = "";
            req.is_valid = false;
        } else {
            req.image_content = doc["img_data"].GetString();
            req.is_valid = true;
        }

        if (!doc.HasMember("req_id") || !doc["req_id"].IsString()) {
            req.task_id = "";
            req.is_valid = false;
        } else {
            req.task_id = doc["req_id"].GetString();
        }
    }

    return req;
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string NanoDetServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std_object_detection_output& model_output) {
    int code = static_cast<int>(status);
    std::string msg = status == StatusCode::OK ? "success" : "fail";

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
 * @param req
 * @param ctx
 */
void NanoDetServer::Impl::do_detection(const det_request& req, seriex_ctx* ctx) {
    // get task receive timestamp
    auto task_receive_ts = Timestamp::now();
    // get resnet model
    NanoDetPtr worker;
    auto find_worker_start_ts = Timestamp::now();

    while (!_m_working_queue.try_dequeue(worker)) {}

    auto worker_found_ts = Timestamp::now() - find_worker_start_ts;

    // construct model input
    base64_input model_input{req.image_content};

    // do classification
    std::string response_body;

    std_object_detection_output model_output;
    if (req.is_valid) {
        auto status = worker->run(model_input, model_output);

        if (status != StatusCode::OK) {
            LOG(ERROR) << "classifier run failed";
        }

        // make response body
        response_body = make_response_body(req.task_id, status, model_output);
    } else {
        response_body = make_response_body("", StatusCode::MODEL_EMPTY_INPUT_IMAGE, model_output);
    }

    while (!_m_working_queue.enqueue(std::move(worker))) {}

    // fill response
    ctx->response->append_output_body(response_body);

    // update task count
    _m_finished_jobs++;
    _m_waiting_jobs--;

    // output log info
    auto task_finish_ts = Timestamp::now();

    auto task_elapse_ts = task_finish_ts - task_receive_ts;

    LOG(INFO) << "task id: " << req.task_id
              << " received at: " << task_receive_ts.to_format_str()
              << " finished at: " << task_finish_ts.to_format_str()
              << " elapse: " << task_elapse_ts << " s"
              << " find work elapse: " << worker_found_ts << " s"
              << " received jobs: " << _m_received_jobs
              << " waiting jobs: " << _m_waiting_jobs
              << " finished jobs: " << _m_finished_jobs
              << " worker queue size: " << _m_working_queue.size_approx();
}

/****************** NanoDetServer Implementation **************/

/***
 *
 */
NanoDetServer::NanoDetServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
NanoDetServer::~NanoDetServer() = default;

/***
 *
 * @param cfg
 * @return
 */
mortred::common::StatusCode NanoDetServer::init(const decltype(toml::parse("")) &config) {
    // init server params
    if (!config.contains("NANODET_DETECTION_SERVER")) {
        LOG(ERROR) << "Config file does not contain NANODET_DETECTION_SERVER section";
        return StatusCode::SERVER_INIT_FAILED;
    }

    toml::value cfg_content = config.at("NANODET_DETECTION_SERVER");
    auto max_connect_nums = static_cast<int>(cfg_content.at("max_connections").as_integer());
    auto peer_resp_timeout = static_cast<int>(cfg_content.at("peer_resp_timeout").as_integer()) * 1000;
    struct WFServerParams params = HTTP_SERVER_PARAMS_DEFAULT;
    params.max_connections = max_connect_nums;
    params.peer_response_timeout = peer_resp_timeout;
    auto&& proc = std::bind(
                      &NanoDetServer::Impl::serve_process, std::cref(this->_m_impl), std::placeholders::_1);
    _m_server = std::make_unique<WFHttpServer>(&params, proc);

    // init _m_impl
    return _m_impl->init(config);
}

/***
 *
 * @param task
 */
void NanoDetServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool NanoDetServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}
}
}
}
