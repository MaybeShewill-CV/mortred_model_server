/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_server_impl.h
* Date: 22-6-30
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H
#define MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H

#include <any>
#include <chrono>
#include <type_traits>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "stl_container/blockingconcurrentqueue.h"
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
#include "common/json_request_parser.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace server {
using jinq::common::Base64;
using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::Md5;
using jinq::common::StatusCode;
using jinq::common::Timestamp;

/***
 * CV 图像 worker 特征：unique_ptr<BaseAiModel<base64_input, OUTPUT>> 走基类默认 do_work；
 * 其他 worker（如 LLM）必须覆写 do_work。
 */
template <typename WORKER>
struct is_cv_worker : std::false_type {};

template <typename INPUT, typename OUTPUT>
struct is_cv_worker<std::unique_ptr<jinq::models::BaseAiModel<INPUT, OUTPUT> > > : std::true_type {};

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
    size_t request_size_limit = -1;

protected:
    // init flag
    bool _m_successfully_initialized = false;
    // task count
    std::atomic<size_t> _m_received_jobs{0};
    std::atomic<size_t> _m_finished_jobs{0};
    std::atomic<size_t> _m_waiting_jobs{0};
    // worker queue
    moodycamel::BlockingConcurrentQueue<WORKER> _m_working_queue;
    // model run timeout
    int _m_model_run_timeout = 500; // ms
    // server uri
    std::string _m_server_uri;

protected:
    struct series_ctx {
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

    /***
     * 通用任务请求：由各服务自行解析并填充 payload（CV：base64 图像字符串；LLM：Dialog 等）。
     */
    struct task_request {
        std::string task_id;
        std::string session_id;
        bool is_valid = false;
        StatusCode parse_status = StatusCode::OK;
        std::string raw_body;
        std::any payload;
    };

protected:
    /***
     *
     * @param req
     * @return
     */
    virtual task_request parse_task_request(const protocol::HttpRequest* req) {
        std::string req_body = protocol::HttpUtil::decode_chunked_body(req);
        auto parsed = jinq::common::parse_json_request(req_body);

        task_request result;
        result.task_id = parsed.task_id;
        result.is_valid = parsed.is_valid;
        result.parse_status = parsed.parse_status;
        result.raw_body = req_body;
        result.payload = parsed.image_content;
        return result;
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
     * 自定义扩展端点钩子：URI 不属于 welcome/hello/model 时调用，
     * 返回 true 表示已处理，false 则回 404。
     * @param task
     * @return
     */
    virtual bool handle_custom_endpoint(WFHttpTask* task) {
        return false;
    }

    /***
     *
     * @param req
     * @param ctx
     */
    virtual void do_work(const task_request& req, series_ctx* ctx);

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
        auto task_req = parse_task_request(req);
        _m_waiting_jobs++;
        _m_received_jobs++;
        // init series work
        auto* series = series_of(task);
        auto* ctx = new series_ctx;
        ctx->response = resp;
        series->set_context(ctx);
        // do model work
        auto&& go_proc = std::bind(&BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work, this, std::placeholders::_1, std::placeholders::_2);
        WFGoTask* serve_task = nullptr;
        if (_m_model_run_timeout <= 0) {
            serve_task = WFTaskFactory::create_go_task(_m_server_uri, go_proc, task_req, ctx);
        } else {
            serve_task = WFTaskFactory::create_timedgo_task(
                0, _m_model_run_timeout * 1e6, _m_server_uri, go_proc, task_req, ctx);
        }
        auto&& go_proc_cb = std::bind(&BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work_cb, this, serve_task);
        serve_task->set_callback(go_proc_cb);
        *series << serve_task;
        WFCounterTask* counter = WFTaskFactory::create_counter_task("release_ctx", 1, [](const WFCounterTask* task){
            delete (series_ctx*)series_of(task)->get_context();
        });
        *series << counter;
        return;
    }
    // not found valid url
    else {
        if (handle_custom_endpoint(task)) {
            return;
        }
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
    const BaseAiServerImpl::task_request& req,
    BaseAiServerImpl::series_ctx* ctx) {
    // get model worker
    WORKER worker;
    auto find_worker_start_ts = Timestamp::now();

    if (_m_model_run_timeout > 0) {
        // 有界等待：等 worker 也计入模型超时预算，形成背压
        if (!_m_working_queue.wait_dequeue_timed(
                worker, std::chrono::milliseconds(_m_model_run_timeout))) {
            ctx->model_run_status = StatusCode::MODEL_RUN_TIMEOUT;
            ctx->task_finished_ts = Timestamp::now().to_format_str();
            // 关键：提前退出也必须恰好计一次 release_ctx，否则 ctx 泄漏
            WFTaskFactory::count_by_name("release_ctx");
            return;
        }
    } else {
        // model_run_timeout <= 0 表示不设超时，用无界阻塞等待
        _m_working_queue.wait_dequeue(worker);
    }
    ctx->find_worker_time_consuming = (Timestamp::now() - find_worker_start_ts) * 1000;

    // get task receive timestamp
    ctx->task_id = req.task_id;
    ctx->is_task_req_valid = req.is_valid;
    auto task_receive_ts = Timestamp::now();
    ctx->task_received_ts = task_receive_ts.to_format_str();

    // construct model input: 默认实现为 CV 图像路径（payload 为 base64 字符串）
    models::io_define::common_io::base64_input model_input;
    StatusCode status = StatusCode::OK;
    if (req.is_valid) {
        if constexpr (is_cv_worker<WORKER>::value) {
            try {
                model_input.input_image_content = std::any_cast<std::string>(req.payload);
                status = worker->run(model_input, ctx->model_output);
            } catch (const std::bad_any_cast&) {
                status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
            }
        } else {
            // 非 CV worker 必须覆写 do_work
            status = StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        if (status != StatusCode::OK) {
            LOG(ERROR) << "worker run failed";
        }
    } else {
        status = req.parse_status;
    }
    ctx->model_run_status = status;

    // restore worker queue
    _m_working_queue.enqueue(std::move(worker));

    // update ctx
    auto task_finish_ts = Timestamp::now();
    ctx->task_finished_ts = task_finish_ts.to_format_str();
    ctx->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000;
    WFTaskFactory::count_by_name("release_ctx");
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
    auto* ctx = (series_ctx*)series_of(task)->get_context();

    // fill response
    StatusCode status;

    if (state != WFT_STATE_SUCCESS) {
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
    // WFTaskFactory::count_by_name("release_ctx");
}
}
}


#endif //MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H
