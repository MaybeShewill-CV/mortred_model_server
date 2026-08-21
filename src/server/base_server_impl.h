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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <utility>
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
#include "common/http_response.h"
#include "common/json_request_parser.h"
#include "common/request_size_limit.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "server/prometheus_metrics.h"
#include "server/http_status.h"
#include "server/openapi_doc.h"
#include "server/rate_limiter.h"
#include "server/server_config_schema.h"

namespace jinq {
namespace server {
using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::common::Timestamp;
using jinq::common::k_default_request_size_limit_mb;

/***
 * Parse and validate the worker count.
 * Missing (value_or falls back to 0), zero or negative are config errors — an
 * empty worker queue would hang do_work's unbounded wait_dequeue forever, so
 * return -1 and let the caller refuse to start.
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

// monotonic clock in milliseconds: immune to wall-clock adjustments, used
// only for duration spans (stuck-worker detection)
inline int64_t monotonic_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Generate a lightweight request id when the client does not provide one.
inline std::string generate_req_id() {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           now.time_since_epoch())
                           .count();
    static std::atomic<uint64_t> seq{0};
    const uint64_t unique = static_cast<uint64_t>(nanos) ^
                            (static_cast<uint64_t>(seq.fetch_add(1)) << 32);
    char buf[32] = {0};
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(unique));
    return std::string(buf);
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
     * web_console's SIGINT -> SIGKILL fallback), not here. The drain only runs
     * when init succeeded: the worker watermark is committed at the end of a
     * successful init, so a partially-filled queue from a failed init would
     * otherwise spin forever — on failure the queue destructor releases the
     * remaining workers itself. Residual note: a go task popped by the
     * executor but preempted before its first queue access is not observable
     * through the queue; this microsecond-level window is mitigated by
     * stop()/wait_finish() preceding destruction in all callers.
     */
    virtual ~BaseAiServerImpl() {
        // only a fully initialized server can possibly hold exactly
        // _m_worker_nums workers: on init failure the queue may hold a
        // partial set (the watermark is committed only after the queue is
        // fully filled), and the queue destructor releases those workers
        // itself — draining here would spin forever
        if (!_m_successfully_initialized) {
            return;
        }
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

    // stuck-worker detection: each failed full-timeout queue wait proves the
    // queue was empty for a whole timeout, but concurrent waits OVERLAP in
    // time when requests queue up, so the failure count alone does not prove
    // duration (a slow-but-healthy worker can rack up K failures in ~1x
    // timeout). Therefore the alarm needs both: K consecutive failures AND a
    // streak spanning >= K x timeout from the first failure — only then has
    // the checked-out worker been gone at least that long. A successful
    // dequeue resets both.
    enum class StuckWorkerAction { LOG, EXIT };
    std::atomic<int> _m_consecutive_wait_timeouts{0};
    std::atomic<int64_t> _m_first_wait_timeout_ms{0};   // first failure of the streak
    StuckWorkerAction _m_stuck_worker_action = StuckWorkerAction::LOG;
    int _m_stuck_worker_threshold_times = 3;
    // server uri
    std::string _m_server_uri;
    // bearer token auth (empty = disabled)
    std::string _m_auth_token;
    // max requests per second per client IP (<= 0 = disabled)
    int _m_rate_limit_qps = 0;
    FixedWindowRateLimiter _m_rate_limiter{0};
    PrometheusMetrics _m_metrics;

protected:
    /***
     * Parse auth and rate-limit config (auth_token / rate_limit_qps).
     * Fail-closed: non-loopback listeners must configure auth_token or refuse
     * to start.
     */
    StatusCode parse_server_security_config(const toml::table& server_section);

    /***
     * Parse common server-section config: 5 server params + request size limit
     * normalization + auth/rate-limit security config.
     */
    StatusCode parse_common_server_config(const toml::table& server_section);

    /***
     * Get the client IP; returns "" on failure.
     */
    static std::string peer_ip_of(const WFHttpTask* task);

    /***
     * Read the Authorization request header.
     */
    static std::string authorization_header_of(const protocol::HttpRequest* req);

    /***
     * Read any request header (name lookup is case-insensitive).
     */
    static std::string header_value_of(const protocol::HttpRequest* req,
                                       const std::string& target_name);

    /***
     * Whether Content-Type is acceptable (application/json, ignoring params and case).
     */
    static bool is_json_content_type(const std::string& content_type);

    /***
     * Unified 401 / 429 responses.
     */
    static void reply_unauthorized(WFHttpTask* task);
    static void reply_rate_limited(WFHttpTask* task);

    /***
     * Unified JSON response exit: sets HTTP status, Content-Type, X-Request-ID.
     */
    static void reply_json(protocol::HttpResponse* resp,
                           const std::string& req_id,
                           StatusCode status,
                           rapidjson::Document&& data) {
        resp->set_status_code(std::to_string(http_status_of(status)).c_str());
        resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
        resp->add_header_pair("X-Request-ID", req_id.c_str());
        resp->add_header_pair("Cache-Control", "no-store");

        jinq::common::HttpResponse http_resp;
        http_resp.req_id = req_id;
        http_resp.code = jinq::common::to_underlying(status);
        http_resp.msg = status == StatusCode::OK
                            ? "success"
                            : jinq::common::status_code_to_str(status);
        http_resp.data = std::move(data);

        auto body = jinq::common::build_response_body(http_resp);
        resp->append_output_body(body.data(), body.size());
    }

    static void reply_json(WFHttpTask* task,
                           const std::string& req_id,
                           StatusCode status,
                           rapidjson::Document&& data) {
        reply_json(task->get_resp(), req_id, status, std::move(data));
    }

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
     * Task request: parse_task_request parses the base64 image content and the
     * caller trace id.
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
    virtual void fill_response_data(
        rapidjson::Document::AllocatorType& allocator,
        rapidjson::Document& data,
        const StatusCode& status,
        const MODEL_OUTPUT& model_output) = 0;

    /***
     * Custom extension endpoint hook: called when the URI is not
     * welcome/hello/model. Return true if handled, false to reply 404.
     * @param task
     * @return
     */
    virtual bool handle_custom_endpoint(WFHttpTask* task) {
        (void)task;
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
    _m_metrics.set_model(_m_server_uri);
    // rate limit: all requests on the port (incl. health checks) count per client IP
    if (_m_rate_limit_qps > 0 && !_m_rate_limiter.allow(peer_ip_of(task))) {
        _m_metrics.inc_http_requests(task->get_req()->get_method(), "429");
        reply_rate_limited(task);
        return;
    }
    // auth: health/metadata endpoints stay public, others require a Bearer Token
    const char* request_uri = task->get_req()->get_request_uri();
    const char* request_method = task->get_req()->get_method();
    bool is_health_endpoint = strcmp(request_uri, "/welcome") == 0 ||
                              strcmp(request_uri, "/hello_world") == 0 ||
                              strcmp(request_uri, "/healthz") == 0 ||
                              strcmp(request_uri, "/ready") == 0 ||
                              strcmp(request_uri, "/metrics") == 0 ||
                              strcmp(request_uri, "/openapi.json") == 0;
    if (!is_health_endpoint &&
        !jinq::common::is_bearer_authorized(
            authorization_header_of(task->get_req()), _m_auth_token)) {
        _m_metrics.inc_http_requests(request_method, "401");
        reply_unauthorized(task);
        return;
    }

    // health / readiness endpoints
    if (strcmp(request_uri, "/healthz") == 0) {
        rapidjson::Document data;
        reply_json(task, "", StatusCode::OK, std::move(data));
        return;
    }
    if (strcmp(request_uri, "/ready") == 0) {
        const bool ready = _m_successfully_initialized && _m_working_queue.size_approx() > 0;
        _m_metrics.set_ready(ready);
        rapidjson::Document data;
        if (ready) {
            reply_json(task, "", StatusCode::OK, std::move(data));
        } else {
            reply_json(task, "", StatusCode::NOT_READY, std::move(data));
        }
        return;
    }

    if (strcmp(request_uri, "/metrics") == 0) {
        auto* resp = task->get_resp();
        resp->set_status_code("200");
        resp->add_header_pair("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        auto body = _m_metrics.render();
        resp->append_output_body(body.data(), body.size());
        return;
    }
    if (strcmp(request_uri, "/openapi.json") == 0) {
        auto* resp = task->get_resp();
        resp->set_status_code("200");
        resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
        resp->add_header_pair("Cache-Control", "no-store");
        resp->append_output_body(k_openapi_doc_json.data(), k_openapi_doc_json.size());
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
        const char* request_method = task->get_req()->get_method();
        if (strcmp(request_method, "POST") != 0) {
            _m_metrics.inc_http_requests(request_method, "405");
            task->get_resp()->add_header_pair("Allow", "POST");
            rapidjson::Document data;
            reply_json(task, "", StatusCode::METHOD_NOT_ALLOWED, std::move(data));
            return;
        }
        // parse request body
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        // 415: model endpoints require application/json (missing is also rejected)
        if (!is_json_content_type(header_value_of(req, "content-type"))) {
            _m_metrics.inc_http_requests(request_method, "415");
            rapidjson::Document data;
            reply_json(task, "", StatusCode::UNSUPPORTED_MEDIA_TYPE, std::move(data));
            return;
        }
        // 413: reject when the declared Content-Length exceeds the limit (chunked is capped by the workflow layer)
        const std::string content_length_str = header_value_of(req, "content-length");
        if (!content_length_str.empty()) {
            char* end = nullptr;
            const unsigned long long declared =
                std::strtoull(content_length_str.c_str(), &end, 10);
            if (end != content_length_str.c_str() && *end == '\0' &&
                declared > _m_request_size_limit * 1024ULL * 1024ULL) {
                _m_metrics.inc_http_requests(request_method, "413");
                rapidjson::Document data;
                reply_json(task, "", StatusCode::REQUEST_ENTITY_TOO_LARGE, std::move(data));
                return;
            }
        }
        auto task_req = parse_task_request(req);
        if (task_req.task_id.empty()) {
            task_req.task_id = generate_req_id();
        }
        _m_waiting_jobs++;
        _m_received_jobs++;
        _m_metrics.inc_received_jobs();
        // init series work: all per-request state rides inside the go task
        // object (functor members + callback closure members), so the
        // framework's task lifetime frees it on every path — no manual
        // release exists anywhere, including cancel before dispatch.
        auto* series = series_of(task);
        request_meta meta;
        meta.task_id = task_req.task_id;
        meta.is_task_req_valid = task_req.is_valid;
        meta.task_received_ts = Timestamp::now().to_format_str();
        go_task_functor functor{this, std::move(task_req), {}};
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
            _m_metrics.inc_finished_jobs();
            _m_waiting_jobs--;
        });
        return;
    }
    // not found valid url
    else {
        if (handle_custom_endpoint(task)) {
            return;
        }
        rapidjson::Document data;
        _m_metrics.inc_http_requests(request_method, "404");
        reply_json(task, "", StatusCode::NOT_FOUND, std::move(data));
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
        // bounded wait: waiting for a worker counts toward the model timeout budget (backpressure)
        if (!_m_working_queue.wait_dequeue_timed(
                worker, std::chrono::milliseconds(_m_model_run_timeout))) {
            result->model_run_status = StatusCode::MODEL_RUN_TIMEOUT;
            result->task_finished_ts = Timestamp::now().to_format_str();
            // span-based stuck judgment: K consecutive failures are necessary
            // but not sufficient (overlapping waits make the count alone lie);
            // require the streak to span >= K x timeout so a slow-but-healthy
            // worker under queue pressure cannot trip the alarm
            int consecutive = _m_consecutive_wait_timeouts.fetch_add(1) + 1;
            if (consecutive == 1) {
                _m_first_wait_timeout_ms.store(monotonic_ms());
            }
            int64_t first = _m_first_wait_timeout_ms.load();
            int64_t span_ms = first > 0 ? monotonic_ms() - first : 0;
            int64_t threshold_ms = static_cast<int64_t>(_m_stuck_worker_threshold_times) *
                                   _m_model_run_timeout;
            if (consecutive >= _m_stuck_worker_threshold_times && span_ms >= threshold_ms) {
                if (_m_stuck_worker_action == StuckWorkerAction::EXIT) {
                    LOG(FATAL) << "worker wait timed out " << consecutive
                               << " times in a row with an empty queue, spanning "
                               << span_ms << " ms (>= " << threshold_ms
                               << " ms): worker stuck, exiting for supervisor restart";
                } else if (consecutive == _m_stuck_worker_threshold_times) {
                    LOG(ERROR) << "worker stuck: " << consecutive
                               << " consecutive full-timeout waits spanning " << span_ms << " ms";
                }
            }
            return;
        }
    } else {
        // model_run_timeout <= 0 means no timeout: use an unbounded blocking wait
        _m_working_queue.wait_dequeue(worker);
    }
    // a successful dequeue means a worker is back: reset the stuck streak
    _m_consecutive_wait_timeouts.store(0);
    _m_first_wait_timeout_ms.store(0);
    result->find_worker_time_consuming = (Timestamp::now() - find_worker_start_ts) * 1000;
    _m_metrics.observe_queue_wait_ms(result->find_worker_time_consuming);

    // local timestamps are for duration stats only; metadata like task_id and
    // task_received_ts is written by serve_process into the closure-carried
    // request_meta, which this routine never touches
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
    // compute before observing: the old order observed the pre-assignment 0,
    // so the histogram always recorded 0 (review leftover #1)
    result->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000;
    _m_metrics.observe_inference_duration_ms(result->worker_run_time_consuming);
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
    rapidjson::Document data;
    std::string task_finished_ts;
    double worker_run_time_consuming = 0;
    double find_worker_time_consuming = 0;

    if (state != WFT_STATE_SUCCESS) {
        // timeout: do_work may still be running detached (or may not have
        // started — user_data may be NULL), so read the closure-carried
        // metadata only; the routine never writes it, hence no race
        status = StatusCode::MODEL_RUN_TIMEOUT;
        task_id = meta.is_task_req_valid ? meta.task_id : "";
    } else {
        // success: the routine has returned (routine -> handle -> done ->
        // callback, workflow guarantees happens-before) and published its ctx
        // address at entry, so every field is safe to read
        auto* result = static_cast<go_result*>(task->user_data);
        status = result->model_run_status;
        task_id = meta.is_task_req_valid ? meta.task_id : "";
        // contract: fill data only on success, keep data:null on error
        if (status == StatusCode::OK) {
            fill_response_data(data.GetAllocator(), data, status, result->model_output);
        }
        task_finished_ts = result->task_finished_ts;
        worker_run_time_consuming = result->worker_run_time_consuming;
        find_worker_time_consuming = result->find_worker_time_consuming;
    }
    reply_json(resp, task_id, status, std::move(data));

    _m_metrics.inc_http_requests("POST", std::to_string(http_status_of(status)));
    _m_metrics.observe_http_duration_ms("POST", std::to_string(http_status_of(status)),
                                        worker_run_time_consuming + find_worker_time_consuming);
    _m_metrics.inc_inference_requests(std::to_string(jinq::common::to_underlying(status)));
    if (status == StatusCode::OK) {
        _m_metrics.inc_inference_success();
    } else {
        _m_metrics.inc_inference_failure();
    }
    _m_metrics.set_workers_available(_m_working_queue.size_approx());
    _m_metrics.set_workers_busy(_m_worker_nums > _m_working_queue.size_approx()
                                    ? _m_worker_nums - _m_working_queue.size_approx()
                                    : 0);
    _m_metrics.set_queue_depth(_m_waiting_jobs.load());
    _m_metrics.set_waiting_jobs(_m_waiting_jobs.load());

    if (state != WFT_STATE_SUCCESS) {
        LOG(ERROR) << "task: " << task_id << " model run timeout";
    }

    // output log info (jobs accounting is done in the series callback)
    LOG(INFO) << "req_id=" << task_id
              << " model=" << _m_server_uri
              << " status=" << jinq::common::to_underlying(status)
              << " received_at=" << meta.task_received_ts
              << " finished_at=" << task_finished_ts
              << " queue_wait_ms=" << find_worker_time_consuming
              << " run_ms=" << worker_run_time_consuming
              << " received_jobs=" << _m_received_jobs.load()
              << " finished_jobs=" << _m_finished_jobs.load()
               << " waiting_jobs=" << _m_waiting_jobs.load()
               << " available_workers=" << _m_working_queue.size_approx();
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
    // fail-fast on malformed server config: missing required keys, wrong
    // types and probable typos are rejected instead of silently falling back
    // to defaults (a mistyped `port` / `worker_nums` must not start a server
    // on a random port or with the wrong worker count)
    std::string schema_err;
    std::vector<std::string> schema_warnings;
    if (!validate_server_section(server_section, &schema_err, &schema_warnings)) {
        LOG(ERROR) << "invalid server config: " << schema_err;
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    for (const auto& warning : schema_warnings) {
        LOG(WARNING) << warning;
    }

    _m_max_connection_nums = static_cast<int>(server_section["max_connections"].value_or<int64_t>(0));
    _m_peer_resp_timeout = static_cast<int>(server_section["peer_resp_timeout"].value_or<int64_t>(0)) * 1000;
    _m_compute_threads = static_cast<int>(server_section["compute_threads"].value_or<int64_t>(0));
    _m_handler_threads = static_cast<int>(server_section["handler_threads"].value_or<int64_t>(0));
    if (auto limit = server_section["request_size_limit"].value_or<int64_t>(0); limit > 0) {
        _m_request_size_limit = static_cast<size_t>(limit);
    }
    // per-request model timeout: a missing key keeps the 500 ms default; a
    // non-positive value disables the timeout — allowed, but warned, since a
    // hung model then holds its worker forever, subsequent requests block
    // indefinitely and clients may never receive a response
    _m_model_run_timeout = static_cast<int>(server_section["model_run_timeout"].value_or<int64_t>(500));
    if (_m_model_run_timeout <= 0) {
        LOG(WARNING) << "model_run_timeout <= 0: per-request timeout disabled; a hung model "
                     << "keeps its worker forever, subsequent requests block indefinitely and "
                     << "clients may never receive a response";
    }
    // stuck-worker detection: "log" only reports; "exit" fails fast so an
    // external supervisor restarts the process with fresh workers. The
    // threshold counts consecutive full-timeout queue waits (only meaningful
    // when model_run_timeout > 0)
    auto action = server_section["stuck_worker_action"].value_or<std::string>("log");
    _m_stuck_worker_action = (action == "exit") ? StuckWorkerAction::EXIT
                                                : StuckWorkerAction::LOG;
    _m_stuck_worker_threshold_times = static_cast<int>(
        server_section["stuck_worker_threshold_times"].value_or<int64_t>(3));
    if (_m_stuck_worker_threshold_times <= 0) {
        _m_stuck_worker_threshold_times = 3;   // misconfigured: keep the safe default
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
std::string BaseAiServerImpl<WORKER, MODEL_OUTPUT>::header_value_of(
    const protocol::HttpRequest* req, const std::string& target_name) {
    protocol::HttpHeaderCursor cursor(req);
    protocol::HttpMessageHeader header;
    std::string target = target_name;
    std::transform(target.begin(), target.end(), target.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    while (cursor.next(&header)) {
        std::string name(static_cast<const char*>(header.name), header.name_len);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (name == target) {
            return std::string(static_cast<const char*>(header.value), header.value_len);
        }
    }
    return "";
}

/***
 *
 * @param req
 * @return
 */
template<typename WORKER, typename MODEL_OUTPUT>
std::string BaseAiServerImpl<WORKER, MODEL_OUTPUT>::authorization_header_of(
    const protocol::HttpRequest* req) {
    return header_value_of(req, "authorization");
}

/***
 * Whether Content-Type is acceptable: case-insensitive, ignoring params like charset.
 */
template<typename WORKER, typename MODEL_OUTPUT>
bool BaseAiServerImpl<WORKER, MODEL_OUTPUT>::is_json_content_type(
    const std::string& content_type) {
    std::string ct = content_type;
    std::transform(ct.begin(), ct.end(), ct.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    const auto semi = ct.find(';');
    if (semi != std::string::npos) {
        ct = ct.substr(0, semi);
    }
    // trim whitespace
    size_t b = 0;
    size_t e = ct.size();
    while (b < e && std::isspace(static_cast<unsigned char>(ct[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(ct[e - 1]))) {
        --e;
    }
    return ct.compare(b, e - b, "application/json") == 0;
}

/***
 *
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::reply_unauthorized(WFHttpTask* task) {
    rapidjson::Document data;
    task->get_resp()->add_header_pair("WWW-Authenticate", "Bearer realm=\"Mortred\"");
    reply_json(task, "", StatusCode::UNAUTHORIZED, std::move(data));
}

/***
 *
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::reply_rate_limited(WFHttpTask* task) {
    rapidjson::Document data;
    reply_json(task, "", StatusCode::RATE_LIMITED, std::move(data));
}
}
}


#endif //MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H
