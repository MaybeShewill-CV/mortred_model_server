/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_server_impl.h
* Date: 22-6-30
************************************************/

#ifndef MM_AI_SERVER_BASE_SERVER_IMPL_H
#define MM_AI_SERVER_BASE_SERVER_IMPL_H

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

namespace jinq {
namespace server {
using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::Md5;
using jinq::common::StatusCode;
using jinq::common::Timestamp;

template<typename WORKER, typename MODEL_OUTPUT>
class BaseAiServerImpl {
public:
    /***
    *
    */
    virtual ~BaseAiServerImpl() = default;

    /***
     *
     * @param config
     */
    BaseAiServerImpl() = default;

    /***
    *
    * @param transformer
    */
    BaseAiServerImpl(const BaseAiServerImpl& BaseAiServerImpl) = default;

    /***
     *
     * @param transformer
     * @return
     */
    BaseAiServerImpl& operator=(const BaseAiServerImpl& transformer) = default;

    /***
     *
     * @param cfg
     * @return
     */
    virtual StatusCode init(const decltype(toml::parse(""))& cfg) = 0;

    /***
    *
    * @param task
    */
    virtual void serve_process(WFHttpTask* task);

    /***
     *
     * @return
     */
    virtual bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

public:
    int max_connection_nums = 200;
    int peer_resp_timeout = 15 * 1000;
    int compute_threads = -1;
    int handler_threads = 50;

protected:
    // init flag
    bool _m_successfully_initialized = false;
    // task count
    std::atomic<size_t> _m_received_jobs{0};
    std::atomic<size_t> _m_finished_jobs{0};
    std::atomic<size_t> _m_waiting_jobs{0};
    // worker queue
    moodycamel::ConcurrentQueue<WORKER> _m_working_queue;
    // model run timeout
    int _m_model_run_timeout = 500; // ms
    // server uri
    std::string _m_server_uri;

protected:
    struct seriex_ctx {
        protocol::HttpResponse* response = nullptr;
        StatusCode model_run_status = StatusCode::OK;
        std::string task_id;
        std::string task_received_ts;
        std::string task_finished_ts;
        bool is_task_req_valid = false;
        double worker_run_time_consuming = 0; // ms
        double find_worker_time_consuming = 0; // ms
        MODEL_OUTPUT model_output;
    };

    struct cls_request {
        std::string image_content;
        std::string task_id;
        bool is_valid = true;
    };

protected:
    /***
     *
     * @param req_body
     * @return
     */
     virtual cls_request parse_task_request(const std::string& req_body) {

        rapidjson::Document doc;
        doc.Parse(req_body.c_str());
        cls_request req{};

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
    };

    /***
     *
     * @param task_id
     * @param status
     * @param model_output
     * @return
     */
    virtual std::string make_response_body(
        const std::string& task_id,
        const StatusCode& status,
        const MODEL_OUTPUT& model_output) = 0;

    /***
     *
     * @param req
     * @param ctx
     */
    virtual void do_work(const cls_request& req, seriex_ctx* ctx);

    /***
     *
     * @param task
     */
    virtual void do_work_cb(const WFGoTask* task);
};

/*********** Public Func Sets **************/

/***
 *
 * @tparam WORKER
 * @tparam MODEL_INPUT
 * @tparam MODEL_OUTPUT
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::serve_process(WFHttpTask* task) {
    // welcome message
    if (strcmp(task->get_req()->get_request_uri(), "/welcome") == 0) {
        task->get_resp()->append_output_body("<html>Welcome to jinq ai server</html>");
        return;
    }
    // hello world message
    else if (strcmp(task->get_req()->get_request_uri(), "/hello_world") == 0) {
        task->get_resp()->append_output_body("<html>Hello World !!!</html>");
        return;
    }
    // model service
    else if (strcmp(task->get_req()->get_request_uri(), _m_server_uri.c_str()) == 0) {
        // parse request body
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto cls_task_req = parse_task_request(protocol::HttpUtil::decode_chunked_body(req));
        _m_waiting_jobs++;
        _m_received_jobs++;
        // init series work
        auto* series = series_of(task);
        auto* ctx = new seriex_ctx;
        ctx->response = resp;
        series->set_context(ctx);
        // do model work
        auto&& go_proc = std::bind(&BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work, this, std::placeholders::_1, std::placeholders::_2);
        WFGoTask* serve_task = nullptr;
        if (_m_model_run_timeout <= 0) {
            serve_task = WFTaskFactory::create_go_task(_m_server_uri, go_proc, cls_task_req, ctx);
        } else {
            serve_task = WFTaskFactory::create_timedgo_task(
                0, _m_model_run_timeout * 1e6, _m_server_uri, go_proc, cls_task_req, ctx);
        }
        auto&& go_proc_cb = std::bind(&BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work_cb, this, serve_task);
        serve_task->set_callback(go_proc_cb);
        *series << serve_task;
        WFCounterTask* counter = WFTaskFactory::create_counter_task("release_ctx", 1, [](const WFCounterTask* task){
            delete (seriex_ctx*)series_of(task)->get_context();
        });
        *series << counter;
        return;
    }
    // not found valid url
    else {
        task->get_resp()->append_output_body("<html>404 Not Found</html>");
        return;
    }
}

/***
 *
 * @tparam WORKER
 * @tparam MODEL_INPUT
 * @tparam MODEL_OUTPUT
 * @param req
 * @param ctx
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work(
    const BaseAiServerImpl::cls_request& req,
    BaseAiServerImpl::seriex_ctx* ctx) {

    // get task receive timestamp
    ctx->task_id = req.task_id;
    ctx->is_task_req_valid = req.is_valid;
    auto task_receive_ts = Timestamp::now();
    ctx->task_received_ts = task_receive_ts.to_format_str();

    // get mobilenetv2 model
    WORKER worker;
    auto find_worker_start_ts = Timestamp::now();

    while (!_m_working_queue.try_dequeue(worker)) {}

    ctx->find_worker_time_consuming = (Timestamp::now() - find_worker_start_ts) * 1000;

    // construct model input
    models::io_define::common_io::base64_input model_input{req.image_content};

    // do classification
    StatusCode status;
    if (req.is_valid) {
        status = worker->run(model_input, ctx->model_output);

        if (status != StatusCode::OK) {
            LOG(ERROR) << "worker run failed";
        }
    } else {
        status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    ctx->model_run_status = status;

    // restore worker queue
    while (!_m_working_queue.enqueue(std::move(worker))) {}

    // update ctx
    auto task_finish_ts = Timestamp::now();
    ctx->task_finished_ts = task_finish_ts.to_format_str();
    ctx->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000;
    // WFTaskFactory::count_by_name("release_ctx");
}

/***
 *
 * @tparam WORKER
 * @tparam MODEL_INPUT
 * @tparam MODEL_OUTPUT
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work_cb(const WFGoTask* task) {
    auto state = task->get_state();
    auto* ctx = (seriex_ctx*)series_of(task)->get_context();

    // fill response
    StatusCode status;

    if (state == WFT_STATE_ABORTED) {
        LOG(ERROR) << "task: " << ctx->task_id << " model run timeout";
        status = StatusCode::MODEL_RUN_TIMEOUT;
    } else {
        status = ctx->model_run_status;
    }

    std::string task_id = ctx->is_task_req_valid ? ctx->task_id : "";
    std::string response_body = make_response_body(task_id, status, ctx->model_output);
    ctx->response->append_output_body(std::move(response_body));

    // update task count
    _m_finished_jobs++;
    _m_waiting_jobs--;

    // output log info
    LOG(INFO) << "task id: " << task_id
              << " received at: " << ctx->task_received_ts
              << " finished at: " << ctx->task_finished_ts
              << " elapse: " << ctx->worker_run_time_consuming << " ms"
              << " find work elapse: " << ctx->find_worker_time_consuming << " ms"
              << " received jobs: " << _m_received_jobs
              << " waiting jobs: " << _m_waiting_jobs
              << " finished jobs: " << _m_finished_jobs
              << " worker queue size: " << _m_working_queue.size_approx();
    WFTaskFactory::count_by_name("release_ctx");
}
}
}


#endif //MM_AI_SERVER_BASE_SERVER_IMPL_H
