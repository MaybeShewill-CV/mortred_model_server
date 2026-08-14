/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_server_impl.h
* Date: 22-6-30
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H
#define MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H

#include <algorithm>
#include <any>
#include <cctype>
#include <chrono>
#include <type_traits>

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
    struct series_ctx {
        protocol::HttpResponse* response = nullptr;
        // 元数据：serve_process 创建 ctx 时写入、之后只读；
        // do_work_cb 超时分支/正常分支均可安全读取（无并发写者）
        std::string task_id;
        std::string task_received_ts;
        bool is_task_req_valid = false;
        // 推理字段：只由 do_work 写；do_work_cb 正常分支读取（go 函数结束 ->
        // handle(ref 原子同步) -> callback，happens-before），超时分支不读取
        StatusCode model_run_status = StatusCode::OK;
        std::string task_finished_ts;
        double worker_run_time_consuming = 0; // ms
        double find_worker_time_consuming = 0; // ms
        MODEL_OUTPUT model_output;
        // 推理字段（LLM）：do_work 写会话 cookie，do_work_cb 正常分支写 response header
        std::string session_cookie;
        // 无名 release counter：do_work 与 do_work_cb 各 count 一次（target=2）。
        // 用指针直接 count 而非 count_by_name，避免并发请求下全局同名 counter 串扰
        WFCounterTask* release_counter = nullptr;
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
        // init series work
        auto* series = series_of(task);
        auto* ctx = new series_ctx;
        ctx->response = resp;
        // 元数据在创建时写入、之后只读：超时 detached 场景下 do_work_cb 读取它们
        // 不会与 do_work 的写入竞争
        ctx->task_id = task_req.task_id;
        ctx->is_task_req_valid = task_req.is_valid;
        ctx->task_received_ts = Timestamp::now().to_format_str();
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
        // release counter target=2：do_work 与 do_work_cb 各 count 一次，
        // delete ctx 只发生在"双方都结束"之后。超时 detached 场景下 do_work 是
        // 最后结束的一方，其 count 才触发释放，杜绝 use-after-free。
        // 用无名 counter（指针 count）而非 count_by_name：并发请求各持自己的
        // counter 实例，避免全局同名 counter 相互串扰。
        auto* counter = WFTaskFactory::create_counter_task(2, [](const WFCounterTask* task){
            delete (series_ctx*)series_of(task)->get_context();
        });
        ctx->release_counter = counter;
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
            ctx->release_counter->count();
            return;
        }
    } else {
        // model_run_timeout <= 0 表示不设超时，用无界阻塞等待
        _m_working_queue.wait_dequeue(worker);
    }
    ctx->find_worker_time_consuming = (Timestamp::now() - find_worker_start_ts) * 1000;

    // 局部时间戳仅用于耗时统计；task_received_ts 等元数据已由 serve_process 写入 ctx
    auto task_receive_ts = Timestamp::now();

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
    ctx->release_counter->count();
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

    StatusCode status;
    std::string task_id;
    std::string response_body;
    std::string task_finished_ts;
    double worker_run_time_consuming = 0;
    double find_worker_time_consuming = 0;

    if (state != WFT_STATE_SUCCESS) {
        // 超时：do_work 可能仍在 detached 线程中写推理字段，此处只读元数据
        // （serve_process 写入、之后只读），不读取 model_output/model_run_status
        // /finished_ts 等推理字段，避免数据竞争
        status = StatusCode::MODEL_RUN_TIMEOUT;
        task_id = ctx->is_task_req_valid ? ctx->task_id : "";
        response_body = make_response_body(task_id, status, MODEL_OUTPUT{});
    } else {
        // 成功：do_work 已完成（go 函数结束 -> handle(ref 原子同步) -> callback，
        // workflow 保证 happens-before），安全读取全部字段
        status = ctx->model_run_status;
        task_id = ctx->is_task_req_valid ? ctx->task_id : "";
        response_body = make_response_body(task_id, status, ctx->model_output);
        task_finished_ts = ctx->task_finished_ts;
        worker_run_time_consuming = ctx->worker_run_time_consuming;
        find_worker_time_consuming = ctx->find_worker_time_consuming;
        // LLM 会话 cookie：do_work 结束后统一写 response header（happens-before）
        if (!ctx->session_cookie.empty()) {
            ctx->response->add_header_pair("Set-Cookie", ctx->session_cookie);
        }
    }
    ctx->response->append_output_body(std::move(response_body));

    if (state != WFT_STATE_SUCCESS) {
        LOG(ERROR) << "task: " << task_id << " model run timeout";
    }

    // update task count
    _m_finished_jobs++;
    _m_waiting_jobs--;

    // output log info
    LOG(INFO) << "task id: " << task_id
              << " received at: " << ctx->task_received_ts
              << " finished at: " << task_finished_ts
              << " elapse: " << worker_run_time_consuming << " ms"
              << " find work elapse: " << find_worker_time_consuming << " ms"
              << " received jobs: " << _m_received_jobs
              << " waiting jobs: " << _m_waiting_jobs
              << " finished jobs: " << _m_finished_jobs
              << " worker queue size: " << _m_working_queue.size_approx();

    // 关键（原生 workflow 语义）：do_work 与 do_work_cb 各 count 一次（target=2），
    // delete ctx 只发生在双方都结束之后；超时 detached 场景下 do_work 是最后
    // 结束的一方，其 count 才触发释放，不再有 use-after-free。
    ctx->release_counter->count();
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
