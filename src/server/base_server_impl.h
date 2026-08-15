/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_server_impl.h
* Date: 22-6-30
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H
#define MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

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

#include "common/auth_token.h"
#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/json_request_parser.h"
#include "common/request_size_limit.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "server/rate_limiter.h"

namespace jinq {
namespace server {
using jinq::common::base64;
using jinq::common::cv_utils;
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::Timestamp;
using jinq::common::k_default_request_size_limit_mb;

/***
 * 解析并校验 worker 数量。
 * 缺失（value_or 回落到 0）、0 或负数都是配置错误——空 worker 队列会让
 * do_work 的无界 wait_dequeue 永久挂起，因此返回 -1 由调用方拒绝启动。
 */
inline int parse_worker_nums(const toml::table& server_section) {
    auto worker_nums = static_cast<int>(server_section["worker_nums"].value_or<int64_t>(0));
    if (worker_nums <= 0) {
        LOG(ERROR) << "invalid worker_nums: " << worker_nums
                   << " (missing, zero or negative), requests would hang forever";
        return -1;
    }
    return worker_nums;
}

template<typename WORKER, typename MODEL_OUTPUT>
class BaseAiServerImpl {
public:
    /***
     * drain in-flight go tasks before members are destroyed: wait until every
     * worker is back in the queue (a running do_work holds exactly one worker,
     * and after returning it touches only its task-owned ctx — a member of the
     * go closure, kept alive by the framework's task lifetime). The wait is
     * deliberately unbounded — a hung model keeps its worker forever and the
     * destructor blocks; that is handled by the outer process manager (e.g.
     * web_console's SIGINT -> SIGKILL fallback), not here. Residual note: a go
     * task popped by the executor but preempted before its first queue access
     * is not observable through the queue; this microsecond-level window is
     * mitigated by stop()/wait_finish() preceding destruction in all callers.
     */
    virtual ~BaseAiServerImpl() {
        constexpr int k_poll_ms = 5;
        while (_m_working_queue.size_approx() != _m_worker_nums) {
            std::this_thread::sleep_for(std::chrono::milliseconds(k_poll_ms));
        }
    }

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
    virtual StatusCode init(const toml::table& cfg) = 0;

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
    int _m_max_connection_nums = 200;
    int _m_peer_resp_timeout = 15 * 1000;
    int _m_compute_threads = -1;
    int _m_handler_threads = 50;
    size_t _m_request_size_limit = k_default_request_size_limit_mb;

protected:
    // init flag
    bool _m_successfully_initialized = false;
    // task count
    std::atomic<size_t> _m_received_jobs{0};
    std::atomic<size_t> _m_finished_jobs{0};
    std::atomic<size_t> _m_waiting_jobs{0};
    // worker queue
    moodycamel::BlockingConcurrentQueue<WORKER> _m_working_queue;
    // total worker count: set by the concrete server's init(); the destructor
    // drain waits for the queue to return to this size (all workers home)
    size_t _m_worker_nums = 0;
    // model run timeout
    int _m_model_run_timeout = 500; // ms
    // server uri
    std::string _m_server_uri;
    // bearer token 鉴权（空 = 关闭）
    std::string _m_auth_token;
    // 每客户端 IP 每秒最大请求数（<= 0 = 关闭）
    int _m_rate_limit_qps = 0;
    FixedWindowRateLimiter _m_rate_limiter{0};

protected:
    /***
     * 解析鉴权与限流配置（auth_token / rate_limit_qps）。
     * fail-closed：非回环监听必须配置 auth_token，否则拒绝启动。
     */
    StatusCode parse_server_security_config(const toml::table& server_section);

    /***
     * 解析 server 段公共配置：5 个 server 参数 + 请求体上限归一化 + 鉴权/限流安全配置。
     */
    StatusCode parse_common_server_config(const toml::table& server_section);

    /***
     * 获取客户端 IP，失败返回空串。
     */
    static std::string peer_ip_of(const WFHttpTask* task);

    /***
     * 读取 Authorization 请求头。
     */
    static std::string authorization_header_of(const protocol::HttpRequest* req);

    /***
     * 401 / 429 统一响应。
     */
    static void reply_unauthorized(WFHttpTask* task);
    static void reply_rate_limited(WFHttpTask* task);

protected:
    // inference result: a member of the go task's closure (go_task_functor::ctx),
    // so the framework's task lifetime owns it on every path — normal completion,
    // timeout with the routine still running detached (the timed ref(4) scheme
    // keeps the task alive until the routine returns, WFTaskFactory.inl), cancel
    // while running (the executor keeps popped tasks alive), cancel before
    // dispatch (the task is destroyed together with its members). Written only
    // by do_work; read by the success branch of do_work_cb via task->user_data.
    struct go_result {
        StatusCode model_run_status = StatusCode::OK;
        std::string task_finished_ts;
        double worker_run_time_consuming = 0; // ms
        double find_worker_time_consuming = 0; // ms
        MODEL_OUTPUT model_output;
    };

    // request metadata: bound by value into the go task's callback closure
    // (another task member, freed with the task). Written once by serve_process
    // before dispatch, then read-only: the timeout branch of do_work_cb reads it
    // while do_work may still run detached, but do_work never writes it, so
    // there is no race.
    struct request_meta {
        std::string task_id;
        std::string task_received_ts;
        bool is_task_req_valid = false;
    };

    /***
     * 任务请求：parse_task_request 解析 base64 图像内容与调用方追踪 id。
     */
    struct task_request {
        std::string task_id;
        bool is_valid = false;
        StatusCode parse_status = StatusCode::OK;
        std::string payload;
    };

    // the go routine carrier: the three members live inside the task object
    // (the factory binds this functor into the go closure), so no manual
    // release exists anywhere — task destruction frees all per-request state
    // on every path, including cancel before dispatch.
    struct go_task_functor {
        BaseAiServerImpl* self;
        task_request req;
        go_result ctx;

        void operator()(WFGoTask* task) {
            // publish the address of THIS copy's ctx: do_work_cb, running
            // inside done() while the task is still alive, finds the result
            // here. Taking the address inside operator() keeps the pointer
            // valid even if std::function/std::bind copied the callable.
            task->user_data = &ctx;
            self->do_work(&req, &ctx);
        }
    };

protected:
    /***
     *
     * @param req
     * @return
     */
    task_request parse_task_request(const protocol::HttpRequest* req) {
        std::string req_body = protocol::HttpUtil::decode_chunked_body(req);
        auto parsed = jinq::common::parse_json_request(req_body);

        task_request result;
        result.task_id = parsed.task_id;
        result.is_valid = parsed.is_valid;
        result.parse_status = parsed.parse_status;
        result.payload = parsed.image_content;
        return result;
    }

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
     * @param result
     */
    void do_work(const task_request* req, go_result* result);

    /***
     *
     * @param task
     * @param meta
     * @param resp
     */
    void do_work_cb(WFGoTask* task, const request_meta& meta, protocol::HttpResponse* resp);
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
    // 限流：端口上的所有请求（含健康检查）按客户端 IP 计数
    if (_m_rate_limit_qps > 0 && !_m_rate_limiter.allow(peer_ip_of(task))) {
        reply_rate_limited(task);
        return;
    }
    // 鉴权：/welcome 与 /hello_world 保持公开（供健康检查），其余端点需要 Bearer Token
    const char* request_uri = task->get_req()->get_request_uri();
    bool is_health_endpoint = strcmp(request_uri, "/welcome") == 0 ||
                              strcmp(request_uri, "/hello_world") == 0;
    if (!is_health_endpoint &&
        !jinq::common::is_bearer_authorized(
            authorization_header_of(task->get_req()), _m_auth_token)) {
        reply_unauthorized(task);
        return;
    }
    // welcome message
    if (strcmp(request_uri, "/welcome") == 0) {
        task->get_resp()->append_output_body("<html>Welcome to jinq ai server</html>");
        return;
    }
    // hello world message
    else if (strcmp(request_uri, "/hello_world") == 0) {
        task->get_resp()->append_output_body("<html>Hello World !!!</html>");
        return;
    }
    // model service
    else if (strcmp(request_uri, _m_server_uri.c_str()) == 0) {
        // parse request body
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto task_req = parse_task_request(req);
        _m_waiting_jobs++;
        _m_received_jobs++;
        // init series work: all per-request state rides inside the go task
        // object (functor members + callback closure members), so the
        // framework's task lifetime frees it on every path — no manual
        // release exists anywhere, including cancel before dispatch.
        auto* series = series_of(task);
        request_meta meta;
        meta.task_id = task_req.task_id;
        meta.is_task_req_valid = task_req.is_valid;
        meta.task_received_ts = Timestamp::now().to_format_str();
        go_task_functor functor{this, std::move(task_req)};
        // create the task with a null routine first, so the task pointer can
        // be bound into the go closure (reset_go_task below): the routine then
        // publishes its ctx address through task->user_data for do_work_cb
        WFGoTask* serve_task = nullptr;
        if (_m_model_run_timeout <= 0) {
            serve_task = WFTaskFactory::create_go_task<std::nullptr_t>(_m_server_uri, nullptr);
        } else {
            serve_task = WFTaskFactory::create_timedgo_task<std::nullptr_t>(
                0, static_cast<long>(_m_model_run_timeout * 1e6), _m_server_uri, nullptr);
        }
        auto&& work_cb = std::bind(&BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work_cb,
                                   this, std::placeholders::_1, std::move(meta), resp);
        serve_task->set_callback(work_cb);
        WFTaskFactory::reset_go_task(serve_task, std::move(functor), serve_task);
        *series << serve_task;

        // jobs accounting only: the series callback runs on both normal
        // completion and cancel() (workflow native semantics; dismiss() is
        // never used on this path). On cancel, do_work_cb does not run, so
        // accounting must live here or it would drift. No resource release is
        // needed here — per-request state dies with the task object itself.
        series->set_callback([this](const SeriesWork*) {
            _m_finished_jobs++;
            _m_waiting_jobs--;
        });
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
    const BaseAiServerImpl::task_request* req,
    BaseAiServerImpl::go_result* result) {
    // the result is a member of the go task's closure: the framework keeps the
    // task alive until this routine returns even when detached (timeout) or
    // when the series is cancelled mid-run, so writing it is always safe, and
    // its destruction is handled by the task — nothing to release here
    WORKER worker;
    auto find_worker_start_ts = Timestamp::now();

    if (_m_model_run_timeout > 0) {
        // 有界等待：等 worker 也计入模型超时预算，形成背压
        if (!_m_working_queue.wait_dequeue_timed(
                worker, std::chrono::milliseconds(_m_model_run_timeout))) {
            result->model_run_status = StatusCode::MODEL_RUN_TIMEOUT;
            result->task_finished_ts = Timestamp::now().to_format_str();
            return;
        }
    } else {
        // model_run_timeout <= 0 表示不设超时，用无界阻塞等待
        _m_working_queue.wait_dequeue(worker);
    }
    result->find_worker_time_consuming = (Timestamp::now() - find_worker_start_ts) * 1000;

    // 局部时间戳仅用于耗时统计；task_id/task_received_ts 等元数据由 serve_process
    // 写进回调闭包携带的 request_meta，routine 不接触
    auto task_receive_ts = Timestamp::now();

    // construct model input: base64 image content from the request payload
    models::io_define::common_io::base64_input model_input;
    StatusCode status = StatusCode::OK;
    if (req->is_valid) {
        model_input.input_image_content = req->payload;
        status = worker->run(model_input, result->model_output);
        if (status != StatusCode::OK) {
            LOG(ERROR) << "worker run failed";
        }
    } else {
        status = req->parse_status;
    }
    result->model_run_status = status;

    // restore worker queue: the last member touch of this routine; afterwards
    // only the task-owned result is written, so the destructor's queue drain
    // stays sound
    _m_working_queue.enqueue(std::move(worker));

    auto task_finish_ts = Timestamp::now();
    result->task_finished_ts = task_finish_ts.to_format_str();
    result->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000;
}

/***
 *
 * @tparam WORKER
 * @tparam MODEL_INPUT
 * @tparam MODEL_OUTPUT
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work_cb(
    WFGoTask* task,
    const BaseAiServerImpl::request_meta& meta,
    protocol::HttpResponse* resp) {
    // this callback runs inside done(), i.e. while the go task object — and
    // therefore the meta bound into this closure and the result reachable via
    // task->user_data — is still alive (WFTask.h done(): callback, then delete)
    auto state = task->get_state();

    StatusCode status;
    std::string task_id;
    std::string response_body;
    std::string task_finished_ts;
    double worker_run_time_consuming = 0;
    double find_worker_time_consuming = 0;

    if (state != WFT_STATE_SUCCESS) {
        // timeout: do_work may still be running detached (or may not have
        // started — user_data may be NULL), so read the closure-carried
        // metadata only; the routine never writes it, hence no race
        status = StatusCode::MODEL_RUN_TIMEOUT;
        task_id = meta.is_task_req_valid ? meta.task_id : "";
        response_body = make_response_body(task_id, status, MODEL_OUTPUT{});
    } else {
        // success: the routine has returned (routine -> handle -> done ->
        // callback, workflow guarantees happens-before) and published its ctx
        // address at entry, so every field is safe to read
        auto* result = static_cast<go_result*>(task->user_data);
        status = result->model_run_status;
        task_id = meta.is_task_req_valid ? meta.task_id : "";
        response_body = make_response_body(task_id, status, result->model_output);
        task_finished_ts = result->task_finished_ts;
        worker_run_time_consuming = result->worker_run_time_consuming;
        find_worker_time_consuming = result->find_worker_time_consuming;
    }
    resp->append_output_body(std::move(response_body));

    if (state != WFT_STATE_SUCCESS) {
        LOG(ERROR) << "task: " << task_id << " model run timeout";
    }

    // output log info (jobs accounting is done in the series callback)
    LOG(INFO) << "task id: " << task_id
              << " received at: " << meta.task_received_ts
              << " finished at: " << task_finished_ts
              << " elapse: " << worker_run_time_consuming << " ms"
              << " find work elapse: " << find_worker_time_consuming << " ms"
              << " received jobs: " << _m_received_jobs
              << " waiting jobs: " << _m_waiting_jobs
              << " finished jobs: " << _m_finished_jobs
              << " worker queue size: " << _m_working_queue.size_approx();
}

/***
 *
 * @param server_section
 * @return
 */
template<typename WORKER, typename MODEL_OUTPUT>
StatusCode BaseAiServerImpl<WORKER, MODEL_OUTPUT>::parse_server_security_config(
    const toml::table& server_section) {
    _m_auth_token = server_section["auth_token"].value_or<std::string>("");
    _m_rate_limit_qps = static_cast<int>(server_section["rate_limit_qps"].value_or<int64_t>(0));
    _m_rate_limiter.set_max_qps(_m_rate_limit_qps);

    auto listen_host = server_section["host"].value_or<std::string>("127.0.0.1");
    if (!jinq::common::is_loopback_host(listen_host) && _m_auth_token.empty()) {
        LOG(ERROR) << "refusing to serve on non-loopback host " << listen_host
                   << " without auth_token configured";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    return StatusCode::OK;
}

/***
 *
 * @param server_section
 * @return
 */
template<typename WORKER, typename MODEL_OUTPUT>
StatusCode BaseAiServerImpl<WORKER, MODEL_OUTPUT>::parse_common_server_config(
    const toml::table& server_section) {
    _m_max_connection_nums = static_cast<int>(server_section["max_connections"].value_or<int64_t>(0));
    _m_peer_resp_timeout = static_cast<int>(server_section["peer_resp_timeout"].value_or<int64_t>(0)) * 1000;
    _m_compute_threads = static_cast<int>(server_section["compute_threads"].value_or<int64_t>(0));
    _m_handler_threads = static_cast<int>(server_section["handler_threads"].value_or<int64_t>(0));
    if (auto limit = server_section["request_size_limit"].value_or<int64_t>(0); limit > 0) {
        _m_request_size_limit = static_cast<size_t>(limit);
    }
    return parse_server_security_config(server_section);
}

/***
 *
 * @param task
 * @return
 */
template<typename WORKER, typename MODEL_OUTPUT>
std::string BaseAiServerImpl<WORKER, MODEL_OUTPUT>::peer_ip_of(const WFHttpTask* task) {
    struct sockaddr_storage peer_addr;
    socklen_t addr_len = sizeof(peer_addr);
    if (task->get_peer_addr(reinterpret_cast<struct sockaddr*>(&peer_addr), &addr_len) != 0) {
        return "";
    }
    char ip_buf[INET6_ADDRSTRLEN] = {0};
    if (peer_addr.ss_family == AF_INET) {
        auto* ipv4 = reinterpret_cast<const struct sockaddr_in*>(&peer_addr);
        inet_ntop(AF_INET, &ipv4->sin_addr, ip_buf, sizeof(ip_buf));
    } else if (peer_addr.ss_family == AF_INET6) {
        auto* ipv6 = reinterpret_cast<const struct sockaddr_in6*>(&peer_addr);
        inet_ntop(AF_INET6, &ipv6->sin6_addr, ip_buf, sizeof(ip_buf));
    }
    return std::string(ip_buf);
}

/***
 *
 * @param req
 * @return
 */
template<typename WORKER, typename MODEL_OUTPUT>
std::string BaseAiServerImpl<WORKER, MODEL_OUTPUT>::authorization_header_of(
    const protocol::HttpRequest* req) {
    protocol::HttpHeaderCursor cursor(req);
    protocol::HttpMessageHeader header;
    while (cursor.next(&header)) {
        std::string name(static_cast<const char*>(header.name), header.name_len);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (name == "authorization") {
            return std::string(static_cast<const char*>(header.value), header.value_len);
        }
    }
    return "";
}

/***
 *
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::reply_unauthorized(WFHttpTask* task) {
    auto* resp = task->get_resp();
    resp->set_status_code("401");
    resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
    resp->append_output_body(
        "{\"code\":401,\"msg\":\"unauthorized: missing or invalid bearer token\"}");
}

/***
 *
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::reply_rate_limited(WFHttpTask* task) {
    auto* resp = task->get_resp();
    resp->set_status_code("429");
    resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
    resp->append_output_body("{\"code\":429,\"msg\":\"too many requests\"}");
}
}
}


#endif //MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H
