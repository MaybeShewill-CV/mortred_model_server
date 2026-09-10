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
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "workflow/Workflow.h"

#include "rapidjson/document.h"

#include "common/auth_token.h"
#include "common/file_path_util.h"
#include "common/request_size_limit.h"
#include "common/response_envelope.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "models/backend/param_spec.h"
#include "server/async_endpoints.h"
#include "server/async_job_table.h"
#include "server/backpressure.h"
#include "server/batch_collector.h"
#include "server/http_status.h"
#include "server/http_wire.h"
#include "server/inference_task.h"
#include "server/item_exec.h"
#include "server/openapi_doc.h"
#include "server/sync_request_graph.h"
#include "server/output_options.h"
#include "server/parsed_request.h"
#include "server/prometheus_metrics.h"
#include "server/rate_limiter.h"
#include "server/request_admission.h"
#include "server/server_runtime_config.h"
#include "server/worker_pool.h"

namespace jinq {
namespace server {
using jinq::common::StatusCode;
using jinq::common::Timestamp;
using jinq::common::k_default_request_size_limit_mb;

template<typename WORKER, typename MODEL_OUTPUT>
class BaseAiServerImpl {
public:
    /***
     * drain in-flight go tasks before members are destroyed: wait until every
     * worker is back in the queue (a running do_work or sync-graph item holds
     * exactly one worker; metrics/EWMA are written before that enqueue, so
     * this wait is enough to keep _m_metrics and the pool EWMA alive. After
     * HTTP has replied, a detached go may still hold the lease until its
     * SUCCESS callback checkin). The wait is
     * deliberately unbounded — a hung model keeps its worker forever and the
     * destructor blocks; that is handled by the outer process manager (e.g.
     * mortred-supervisor's SIGINT -> SIGKILL fallback), not here. Model and
     * gateway mains arm ProcessStop so SIGINT/SIGTERM reach server->stop()
     * and this drain; a hung model is still SIGKILL'd after kStopGraceMs.
     * The drain only runs when init succeeded: the worker watermark is
     * committed at the end of a successful init, so a partially-filled queue
     * from a failed init would otherwise spin forever — on failure the queue
     * destructor releases the remaining workers itself. Residual note: a go
     * task popped by the executor but preempted before its first queue access
     * is not observable through the queue; this microsecond-level window is
     * mitigated by stop()/wait_finish() preceding destruction in all callers.
     */
    virtual ~BaseAiServerImpl() {
        // stop the batch runner first: it may hold a worker, and queued
        // entries must be failed before the worker drain below
        _m_batch.stop();
        if (!_m_successfully_initialized) {
            return;
        }
        _m_async_eps.drain();
        _m_workers.drain();
    }

    BaseAiServerImpl() = default;
    BaseAiServerImpl(const BaseAiServerImpl&) = delete;
    BaseAiServerImpl& operator=(const BaseAiServerImpl&) = delete;

    virtual StatusCode init(const toml::table& cfg) = 0;
    virtual void serve_process(WFHttpTask* task);
    virtual bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    }

public:
    int _m_max_connection_nums = 200;
    int _m_peer_resp_timeout = 15 * 1000;
    int _m_compute_threads = -1;
    int _m_handler_threads = 50;
    size_t _m_request_size_limit = k_default_request_size_limit_mb;

protected:
    bool _m_successfully_initialized = false;
    std::atomic<size_t> _m_received_jobs{0};
    std::atomic<size_t> _m_finished_jobs{0};
    std::atomic<size_t> _m_waiting_jobs{0};
    WorkerPool<WORKER> _m_workers;
    int _m_model_run_timeout = 500;
    std::string _m_server_uri;
    std::string _m_auth_token;
    int _m_rate_limit_qps = 0;
    FixedWindowRateLimiter _m_rate_limiter{0};
    PrometheusMetrics _m_metrics;
    int _m_max_queue_depth = 0;
    int _m_max_batch_size = 1;
    size_t _m_max_request_items = 16;
    BatchCollector<WORKER, MODEL_OUTPUT> _m_batch{_m_workers, _m_metrics};
    bool _m_async_enabled = false;
    int _m_async_timeout = 300000;
    std::vector<jinq::models::backend::ParamSpec> _m_param_specs;
    std::string _m_model_name;
    using AsyncTable = AsyncJobTable<MODEL_OUTPUT>;
    AsyncTable _m_async_table;
    AsyncJobEndpoints<MODEL_OUTPUT> _m_async_eps;

protected:
    StatusCode parse_common_server_config(const toml::table& server_section);
    void adopt_worker(WORKER&& worker) {
        _m_workers.adopt(std::move(worker));
    }
    void commit_identity(std::string server_uri, std::string model_name,
                         std::vector<jinq::models::backend::ParamSpec> param_specs) {
        _m_server_uri = std::move(server_uri);
        _m_model_name = std::move(model_name);
        _m_param_specs = std::move(param_specs);
    }
    void commit_workers(size_t worker_nums) {
        _m_workers.commit_watermark(worker_nums);
        _m_successfully_initialized = true;
        if (_m_max_batch_size > 1) {
            _m_batch.start();
        }
    }

    ParsedRequest parse_model_request(const protocol::HttpRequest* req) {
        std::string req_body = protocol::HttpUtil::decode_chunked_body(req);
        return parse_request_envelope(req_body, _m_param_specs);
    }

    virtual void fill_response_data(
        rapidjson::Document::AllocatorType& allocator,
        rapidjson::Document& data,
        const MODEL_OUTPUT& model_output,
        const jinq::server::OutputOptions& options) = 0;

    using InferenceResult = jinq::server::InferenceResult<MODEL_OUTPUT>;
    using InferenceTask = jinq::server::InferenceTask;
    using SyncState = SyncRequestState<WORKER, MODEL_OUTPUT>;

    void async_run_job(const std::string& job_id);
    void schedule_async_job(const std::string& job_id);
    void do_work(InferenceTask* req, InferenceResult* result);
    void schedule_sync_request(WFHttpTask* http_task, InferenceTask req, size_t n_items);
    void start_sync_item(std::shared_ptr<SyncState> state, size_t k);
    void start_sync_batch(std::shared_ptr<SyncState> state);
    void on_sync_item_done(std::shared_ptr<SyncState> state, size_t k,
                           std::shared_ptr<typename SyncState::ItemScratch> work,
                           WFGoTask* task);
    void reply_sync_request(std::shared_ptr<SyncState> state);
    void release_sync_worker(std::shared_ptr<SyncState> state);
    void apply_runtime_config(const ServerRuntimeConfig& cfg);
    void wire_async_endpoints();
};

/*********** Public Func Sets **************/

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::apply_runtime_config(const ServerRuntimeConfig& cfg) {
    _m_max_connection_nums = cfg.max_connections;
    _m_peer_resp_timeout = cfg.peer_resp_timeout;
    _m_compute_threads = cfg.compute_threads;
    _m_handler_threads = cfg.handler_threads;
    _m_request_size_limit = cfg.request_size_limit;
    _m_model_run_timeout = cfg.model_run_timeout;
    _m_auth_token = cfg.auth_token;
    _m_rate_limit_qps = cfg.rate_limit_qps;
    _m_rate_limiter.set_max_qps(_m_rate_limit_qps);
    _m_max_queue_depth = cfg.max_queue_depth;
    _m_max_batch_size = cfg.max_batch_size;
    _m_max_request_items = cfg.max_request_items;
    _m_async_enabled = cfg.async_enabled;
    _m_async_timeout = cfg.async_timeout;
    _m_workers.configure_stuck(
        cfg.stuck_worker_action == "exit" ? StuckWorkerAction::EXIT : StuckWorkerAction::LOG,
        cfg.stuck_worker_threshold_times, cfg.model_run_timeout);
    _m_workers.seed_ewma(cfg.ewma_seed_ms);
    _m_batch.configure(cfg.max_batch_size, cfg.max_batch_delay_ms, cfg.model_run_timeout);
    typename AsyncTable::Config async_cfg;
    async_cfg.max_queue = cfg.async_max_queue;
    async_cfg.job_ttl_ms = cfg.async_job_ttl_ms;
    async_cfg.max_completed = cfg.async_max_completed;
    _m_async_table.configure(async_cfg);
    _m_async_eps.configure(_m_async_enabled, _m_async_timeout, _m_max_request_items);
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::wire_async_endpoints() {
    _m_async_eps.bind(
        &_m_async_table, &_m_metrics, &_m_model_name,
        [this](const protocol::HttpRequest* req) { return parse_model_request(req); },
        [this](rapidjson::Document::AllocatorType& allocator, rapidjson::Document& data,
               const MODEL_OUTPUT& output, const OutputOptions& options) {
            fill_response_data(allocator, data, output, options);
        },
        [this](const std::string& job_id) { schedule_async_job(job_id); });
}

template<typename WORKER, typename MODEL_OUTPUT>
StatusCode BaseAiServerImpl<WORKER, MODEL_OUTPUT>::parse_common_server_config(
    const toml::table& server_section) {
    ServerRuntimeConfig cfg;
    const auto status = parse_server_runtime_config(server_section, cfg);
    if (status != StatusCode::OK) {
        _m_successfully_initialized = false;
        return status;
    }
    apply_runtime_config(cfg);
    wire_async_endpoints();
    return StatusCode::OK;
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::schedule_async_job(const std::string& job_id) {
    auto* go_task = WFTaskFactory::create_go_task(
        "async_job", [this, job_id]() { async_run_job(job_id); });
    go_task->set_callback([this, job_id](WFGoTask*) {
        _m_async_eps.on_job_compute_finished(job_id);
    });
    go_task->start();
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::async_run_job(const std::string& job_id) {
    _m_async_table.transition_running(job_id);
    _m_metrics.inc_async_jobs("running");

    WORKER worker;
    const std::chrono::milliseconds wait = _m_async_timeout > 0
        ? std::chrono::milliseconds(_m_async_timeout)
        : std::chrono::milliseconds(-1);
    const auto ck = _m_workers.checkout(worker, wait);
    if (!ck.ok) {
        _m_async_table.timeout(job_id, "worker wait timeout");
        _m_metrics.inc_async_jobs("timeout");
        _m_metrics.set_async_queue_depth(
            static_cast<size_t>(std::max(0, _m_async_table.queue_depth())));
        return;
    }
    _m_metrics.observe_queue_wait_ms(ck.wait_ms);

    auto req = _m_async_table.take_request(job_id);
    if (!req.has_value()) {
        _m_workers.checkin(std::move(worker));
        _m_async_table.fail(job_id, "job request missing");
        _m_metrics.inc_async_jobs("failed");
        return;
    }

    const auto run_start = Timestamp::now();
    InferenceResult result;
    run_items(worker, *req, &result);
    const auto run_end = Timestamp::now();
    const double run_ms = (run_end - run_start) * 1000.0;
    result.task_finished_ts = run_end.to_format_str();
    result.worker_run_time_consuming = run_ms;
    _m_workers.observe_run_ms(static_cast<int64_t>(run_ms));
    _m_workers.checkin(std::move(worker));

    if (result.model_run_status == StatusCode::OK ||
        result.model_run_status == StatusCode::DEADLINE_EXCEEDED_PARTIAL) {
        _m_async_table.finish(job_id, std::move(result));
        _m_metrics.inc_async_jobs("done");
    } else {
        _m_async_table.fail(job_id, jinq::common::status_code_to_str(result.model_run_status));
        _m_metrics.inc_async_jobs("failed");
    }
    _m_metrics.observe_async_job_duration_ms(run_ms);
    _m_metrics.set_async_queue_depth(
        static_cast<size_t>(std::max(0, _m_async_table.queue_depth())));
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::serve_process(WFHttpTask* task) {
    _m_metrics.set_model(_m_server_uri);
    const char* request_uri = task->get_req()->get_request_uri();
    const char* request_method = task->get_req()->get_method();
    if (request_uri == nullptr) {
        request_uri = "";
    }
    if (request_method == nullptr) {
        request_method = "";
    }
    if (_m_rate_limit_qps > 0 && !_m_rate_limiter.allow(peer_ip_of(task))) {
        _m_metrics.inc_http_requests(request_method, "429");
        reply_rate_limited(task, _m_model_name);
        return;
    }
    bool is_health_endpoint = strcmp(request_uri, "/healthz") == 0 ||
                              strcmp(request_uri, "/ready") == 0 ||
                              strcmp(request_uri, "/openapi.json") == 0;
    const bool is_async_endpoint = _m_async_enabled &&
                                   strncmp(request_uri, "/jobs", 5) == 0 &&
                                   (request_uri[5] == '\0' || request_uri[5] == '/');
    if (!is_health_endpoint &&
        !jinq::common::is_bearer_authorized(
            authorization_header_of(task->get_req()), _m_auth_token)) {
        _m_metrics.inc_http_requests(request_method, "401");
        reply_unauthorized(task, _m_model_name);
        return;
    }

    if (is_async_endpoint) {
        _m_async_eps.handle(task);
        return;
    }

    if (strcmp(request_uri, "/healthz") == 0) {
        reply_status(task, StatusCode::OK, _m_model_name);
        return;
    }
    if (strcmp(request_uri, "/ready") == 0) {
        const bool ready = _m_successfully_initialized && _m_workers.available_approx() > 0;
        _m_metrics.set_ready(ready);
        reply_status(task, ready ? StatusCode::OK : StatusCode::NOT_READY, _m_model_name);
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
    if (strcmp(request_uri, _m_server_uri.c_str()) == 0) {
        if (strcmp(request_method, "POST") != 0) {
            _m_metrics.inc_http_requests(request_method, "405");
            task->get_resp()->add_header_pair("Allow", "POST");
            reply_status(task, StatusCode::METHOD_NOT_ALLOWED, _m_model_name);
            return;
        }
        auto* req = task->get_req();
        const std::string content_type = header_value_of(req, "content-type");
        std::string request_encoding = "json";
        jinq::server::ParsedRequest parsed;
        const auto kind = content_type_kind(content_type);
        if (kind == ContentTypeKind::Json) {
            parsed = parse_model_request(req);
        } else if (kind == ContentTypeKind::RawBody) {
            request_encoding = "raw";
            parsed = parse_raw_request(protocol::HttpUtil::decode_chunked_body(req),
                                       header_value_of(req, "x-request-id"),
                                       header_value_of(req, "x-mortred-params"),
                                       header_value_of(req, "x-mortred-options"),
                                       _m_param_specs);
        } else {
            _m_metrics.inc_http_requests(request_method, "415");
            reply_status(task, StatusCode::UNSUPPORTED_MEDIA_TYPE, _m_model_name);
            return;
        }
        _m_metrics.inc_request_encoding(request_encoding);
        if (declared_body_exceeds(header_value_of(req, "content-length"), _m_request_size_limit)) {
            _m_metrics.inc_http_requests(request_method, "413");
            reply_status(task, StatusCode::REQUEST_ENTITY_TOO_LARGE, _m_model_name);
            return;
        }

        const std::string task_id = parsed.req_id.empty() ? generate_req_id() : parsed.req_id;
        auto reply_reject = [&](StatusCode status,
                                std::vector<jinq::common::ResponseError> errors) {
            _m_metrics.inc_http_requests(request_method, std::to_string(http_status_of(status)));
            _m_metrics.inc_inference_requests(jinq::common::to_underlying(status));
            _m_metrics.inc_inference_failure();
            reply_unified_json(task->get_resp(),
                               unified_rejection(task_id, status, std::move(errors)));
        };
        if (!parsed.is_valid) {
            std::vector<jinq::common::ResponseError> errors;
            errors.reserve(parsed.violations.size());
            for (const auto& violation : parsed.violations) {
                errors.push_back({violation.pointer, violation.message});
            }
            reply_reject(parsed.status, std::move(errors));
            return;
        }
        if (item_count_exceeds(parsed.items.size(), _m_max_request_items)) {
            std::vector<jinq::common::ResponseError> limit_error;
            limit_error.push_back({"/images", "too many items in one request (max " +
                                                    std::to_string(_m_max_request_items) + ")"});
            reply_reject(StatusCode::REQUEST_ITEM_LIMIT, std::move(limit_error));
            return;
        }
        const size_t n_items = parsed.items.size();

        if (queue_would_overflow(_m_waiting_jobs.load(), n_items, _m_max_queue_depth)) {
            _m_metrics.inc_queue_rejected();
            _m_metrics.inc_http_requests(request_method, "429");
            const int retry_after = compute_retry_after_seconds(
                _m_waiting_jobs.load(), _m_workers.ewma_ms(), _m_workers.watermark());
            task->get_resp()->add_header_pair("Retry-After",
                                              std::to_string(retry_after).c_str());
            reply_status(task, StatusCode::RATE_LIMITED, _m_model_name);
            return;
        }

        InferenceTask task_req;
        task_req.task_id = task_id;
        task_req.items = std::move(parsed.items);
        task_req.params = std::move(parsed.params);
        task_req.options = parsed.options;
        if (_m_model_run_timeout > 0) {
            task_req.deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(_m_model_run_timeout);
        }

        _m_waiting_jobs += n_items;
        _m_received_jobs += n_items;
        _m_metrics.inc_received_jobs(n_items);
        schedule_sync_request(task, std::move(task_req), n_items);
        return;
    } else {
        _m_metrics.inc_http_requests(request_method, "404");
        reply_status(task, StatusCode::NOT_FOUND, _m_model_name);
        return;
    }
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work(
    InferenceTask* req, InferenceResult* result) {
    if (_m_max_batch_size > 1) {
        _m_batch.submit_and_wait(req, result);
        return;
    }
    WORKER worker;
    const bool has_deadline = req->deadline != std::chrono::steady_clock::time_point::max();
    std::chrono::milliseconds wait{-1};
    if (has_deadline) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                             req->deadline - std::chrono::steady_clock::now()).count();
        if (remaining < 0) {
            remaining = 0;
        }
        wait = std::chrono::milliseconds(remaining);
    }
    const auto ck = _m_workers.checkout(worker, wait);
    if (!ck.ok) {
        result->model_run_status = StatusCode::MODEL_RUN_TIMEOUT;
        result->task_finished_ts = Timestamp::now().to_format_str();
        result->item_status.assign(req->item_count(), StatusCode::MODEL_RUN_TIMEOUT);
        result->item_outputs.assign(req->item_count(), MODEL_OUTPUT{});
        return;
    }
    result->find_worker_time_consuming = ck.wait_ms;
    _m_metrics.observe_queue_wait_ms(ck.wait_ms);

    const auto task_receive_ts = Timestamp::now();
    run_items(worker, *req, result);

    const auto task_finish_ts = Timestamp::now();
    result->task_finished_ts = task_finish_ts.to_format_str();
    result->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000;
    _m_metrics.observe_inference_duration_ms(result->worker_run_time_consuming);
    _m_workers.observe_run_ms(static_cast<int64_t>(result->worker_run_time_consuming));
    _m_workers.checkin(std::move(worker));
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::schedule_sync_request(
    WFHttpTask* http_task, InferenceTask req, size_t n_items) {
    auto state = std::make_shared<SyncState>();
    state->n_items = n_items;
    state->req = std::move(req);
    state->task_id = state->req.task_id;
    state->resp = http_task->get_resp();
    state->waiter = make_sync_waiter_name(state->req.task_id);
    state->result.options = state->req.options;
    state->result.item_status.assign(n_items, StatusCode::MODEL_RUN_TIMEOUT);
    state->result.item_outputs.assign(n_items, MODEL_OUTPUT{});

    auto* series = series_of(http_task);
    auto* counter = WFTaskFactory::create_counter_task(
        state->waiter, 1, [this, state](WFCounterTask*) { reply_sync_request(state); });
    *series << counter;
    series->set_callback([this, n_items](const SeriesWork*) {
        _m_finished_jobs += n_items;
        _m_metrics.inc_finished_jobs(n_items);
        _m_waiting_jobs -= n_items;
    });

    if (_m_max_batch_size > 1) {
        // Collector wait_until already returns at the request deadline. An
        // outer timer racing that wait would reply with published==0 and drop
        // any items the runner had already filled.
        start_sync_batch(state);
        return;
    }

    start_sync_item(state, 0);
    if (_m_model_run_timeout <= 0) {
        return;
    }
    const int timeout_ms = _m_model_run_timeout;
    const std::string waiter = state->waiter;
    auto* timer = WFTaskFactory::create_timer_task(
        static_cast<time_t>(timeout_ms / 1000),
        static_cast<long>((timeout_ms % 1000) * 1000000L),
        [waiter](WFTimerTask*) { WFTaskFactory::count_by_name(waiter, 1); });
    timer->start();
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::start_sync_batch(
    std::shared_ptr<SyncState> state) {
    const size_t n_items = state->n_items;
    auto* go = WFTaskFactory::create_go_task(_m_server_uri, [this, state]() {
        do_work(&state->req, &state->result);
    });
    go->set_callback([state, n_items](WFGoTask* task) {
        if (task->get_state() == WFT_STATE_SUCCESS) {
            state->published.store(n_items, std::memory_order_release);
            state->find_worker_ms.store(
                static_cast<int64_t>(state->result.find_worker_time_consuming),
                std::memory_order_relaxed);
            state->worker_run_ms.store(
                static_cast<int64_t>(state->result.worker_run_time_consuming),
                std::memory_order_relaxed);
        }
        WFTaskFactory::count_by_name(state->waiter, 1);
    });
    go->start();
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::start_sync_item(
    std::shared_ptr<SyncState> state, size_t k) {
    auto work = std::make_shared<typename SyncState::ItemScratch>();
    auto* go = WFTaskFactory::create_go_task(_m_server_uri, [this, state, k, work]() {
        if (state->replied.load(std::memory_order_acquire)) {
            return;
        }
        if (k == 0) {
            WORKER leased;
            const auto ck = _m_workers.checkout(leased, checkout_wait_for(state->req));
            if (!ck.ok) {
                state->checkout_failed.store(true, std::memory_order_release);
                return;
            }
            state->find_worker_ms.store(static_cast<int64_t>(ck.wait_ms),
                                        std::memory_order_relaxed);
            _m_metrics.observe_queue_wait_ms(ck.wait_ms);
            state->worker = std::move(leased);
            state->has_worker.store(true, std::memory_order_release);
        }
        if (state->replied.load(std::memory_order_acquire) ||
            state->checkout_failed.load(std::memory_order_acquire) ||
            !state->has_worker.load(std::memory_order_acquire)) {
            return;
        }
        if (state->req.deadline != std::chrono::steady_clock::time_point::max() &&
            std::chrono::steady_clock::now() >= state->req.deadline) {
            return;
        }
        const auto t0 = Timestamp::now();
        work->status = run_one(state->worker, state->req, k, &work->output);
        work->ran = true;
        work->run_ms = (Timestamp::now() - t0) * 1000.0;
    });
    go->set_callback([this, state, k, work](WFGoTask* task) {
        on_sync_item_done(state, k, work, task);
    });
    go->start();
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::on_sync_item_done(
    std::shared_ptr<SyncState> state, size_t k,
    std::shared_ptr<typename SyncState::ItemScratch> work, WFGoTask* task) {
    bool observed = false;
    const auto observe_run = [this, work, &observed]() {
        if (observed || !work->ran) {
            return;
        }
        _m_metrics.observe_inference_duration_ms(work->run_ms);
        _m_workers.observe_run_ms(static_cast<int64_t>(work->run_ms));
        observed = true;
    };
    const auto finish_and_count = [this, state, &observe_run]() {
        observe_run();
        release_sync_worker(state);
        WFTaskFactory::count_by_name(state->waiter, 1);
    };

    if (state->replied.load(std::memory_order_acquire)) {
        observe_run();
        release_sync_worker(state);
        return;
    }
    if (task->get_state() != WFT_STATE_SUCCESS ||
        state->checkout_failed.load(std::memory_order_acquire) || !work->ran) {
        finish_and_count();
        return;
    }

    state->result.item_outputs[k] = std::move(work->output);
    state->result.item_status[k] = work->status;
    state->published.store(k + 1, std::memory_order_release);
    state->worker_run_ms.fetch_add(static_cast<int64_t>(work->run_ms),
                                   std::memory_order_relaxed);
    observe_run();

    if (state->replied.load(std::memory_order_acquire)) {
        release_sync_worker(state);
        return;
    }

    const bool past_deadline =
        state->req.deadline != std::chrono::steady_clock::time_point::max() &&
        std::chrono::steady_clock::now() >= state->req.deadline;
    if (k + 1 < state->n_items && !past_deadline) {
        start_sync_item(state, k + 1);
        return;
    }
    finish_and_count();
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::release_sync_worker(
    std::shared_ptr<SyncState> state) {
    if (!state->has_worker.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    _m_workers.checkin(std::move(state->worker));
}

template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::reply_sync_request(
    std::shared_ptr<SyncState> state) {
    if (state->replied.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    const size_t n = state->published.load(std::memory_order_acquire);
    state->result.find_worker_time_consuming =
        static_cast<double>(state->find_worker_ms.load(std::memory_order_relaxed));
    state->result.worker_run_time_consuming =
        static_cast<double>(state->worker_run_ms.load(std::memory_order_relaxed));
    state->result.task_finished_ts = Timestamp::now().to_format_str();
    auto snap = assemble_published(state->result, n, state->n_items);
    auto unified = inference_result_to_unified(
        state->task_id, _m_model_name, snap,
        [this](rapidjson::Document::AllocatorType& allocator, rapidjson::Document& data,
               const MODEL_OUTPUT& output, const OutputOptions& options) {
            fill_response_data(allocator, data, output, options);
        });
    const StatusCode status = snap.model_run_status;
    unified.status = jinq::common::to_underlying(status);
    unified.status_str = jinq::common::status_code_to_str(status);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "worker run failed with status " << jinq::common::to_underlying(status);
    }
    reply_unified_json(state->resp, unified);

    size_t ok_items = 0;
    for (const auto& item : unified.results) {
        if (item.status == jinq::common::to_underlying(StatusCode::OK)) {
            ++ok_items;
        }
    }
    const double http_ms =
        snap.worker_run_time_consuming + snap.find_worker_time_consuming;
    _m_metrics.inc_http_requests("POST", std::to_string(http_status_of(status)));
    _m_metrics.observe_http_duration_ms("POST", std::to_string(http_status_of(status)), http_ms);
    for (const auto& item : unified.results) {
        _m_metrics.inc_inference_requests(item.status);
    }
    _m_metrics.inc_inference_success(ok_items);
    _m_metrics.inc_inference_failure(unified.results.size() - ok_items);
    _m_metrics.set_workers_available(_m_workers.available_approx());
    _m_metrics.set_workers_busy(_m_workers.watermark() > _m_workers.available_approx()
                                    ? _m_workers.watermark() - _m_workers.available_approx()
                                    : 0);
    _m_metrics.set_queue_depth(_m_waiting_jobs.load());
    _m_metrics.set_waiting_jobs(_m_waiting_jobs.load());
}

}
}

#endif //MORTRED_MODEL_SERVER_BASE_SERVER_IMPL_H
