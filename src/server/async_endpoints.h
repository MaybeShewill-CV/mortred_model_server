/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: async_endpoints.h
* Date: 26-9-9
************************************************/

// HTTP adapter for /jobs. Owns waiter registration, inflight counters and
// the four endpoints. Does not checkout workers: async_run_job stays on the
// orchestrator and is started via schedule_job. Wake waiters only after the
// compute go-task returns (named counters remember a count that races ahead
// of the waiter).

#ifndef MORTRED_SERVER_ASYNC_ENDPOINTS_H
#define MORTRED_SERVER_ASYNC_ENDPOINTS_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/Workflow.h"

#include "common/response_envelope.h"
#include "common/status_code.h"
#include "server/async_job_table.h"
#include "server/http_wire.h"
#include "server/inference_task.h"
#include "server/item_exec.h"
#include "server/output_options.h"
#include "server/parsed_request.h"
#include "server/prometheus_metrics.h"
#include "server/request_admission.h"

namespace jinq {
namespace server {
using jinq::common::StatusCode;

template <typename MODEL_OUTPUT>
class AsyncJobEndpoints {
public:
    using AsyncTable = AsyncJobTable<MODEL_OUTPUT>;
    using FillFn = std::function<void(rapidjson::Document::AllocatorType&,
                                      rapidjson::Document&,
                                      const MODEL_OUTPUT&,
                                      const OutputOptions&)>;
    using ScheduleFn = std::function<void(const std::string&)>;
    using ParseFn = std::function<ParsedRequest(const protocol::HttpRequest*)>;

    AsyncJobEndpoints() = default;
    AsyncJobEndpoints(const AsyncJobEndpoints&) = delete;
    AsyncJobEndpoints& operator=(const AsyncJobEndpoints&) = delete;

    void bind(AsyncTable* table, PrometheusMetrics* metrics, std::string* model_name,
              ParseFn parse, FillFn fill, ScheduleFn schedule) {
        _table = table;
        _metrics = metrics;
        _model_name = model_name;
        _parse = std::move(parse);
        _fill = std::move(fill);
        _schedule = std::move(schedule);
    }

    void configure(bool enabled, int timeout_ms, size_t max_request_items) {
        _enabled = enabled;
        _timeout_ms = timeout_ms;
        _max_request_items = max_request_items;
    }

    bool enabled() const {
        return _enabled;
    }

    void handle(WFHttpTask* task) {
        const char* raw_uri = task->get_req()->get_request_uri();
        const char* raw_method = task->get_req()->get_method();
        const std::string uri = raw_uri == nullptr ? "" : raw_uri;
        const std::string method = raw_method == nullptr ? "" : raw_method;
        const std::string path = uri.substr(0, uri.find('?'));

        if (path == "/jobs" && method == "POST") {
            handle_async_submit(task);
        } else if (path.rfind("/jobs/", 0) == 0) {
            const std::string rest = path.substr(6);
            const auto slash = rest.find('/');
            if (slash == std::string::npos) {
                if (method != "GET") {
                    reply_async_error(task, 405, "method not allowed");
                    return;
                }
                handle_async_status(task, rest);
            } else {
                const std::string id = rest.substr(0, slash);
                const std::string action = rest.substr(slash + 1);
                if (action == "wait" && method == "GET") {
                    handle_async_wait(task, id, uri);
                } else if (action == "result" && method == "GET") {
                    handle_async_result(task, id);
                } else {
                    reply_async_error(task, 404, "not found");
                }
            }
        } else {
            reply_async_error(task, 404, "not found");
        }
    }

    void on_job_compute_finished(const std::string& job_id) {
        wake_async_waiters(job_id);
        _inflight.fetch_sub(1, std::memory_order_acq_rel);
    }

    void drain() {
        constexpr int k_poll_ms = 5;
        while (_inflight.load(std::memory_order_acquire) != 0 ||
               _wait_inflight.load(std::memory_order_acquire) != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(k_poll_ms));
        }
    }

private:
    static int parse_wait_timeout_ms(const std::string& uri) {
        int timeout_ms = 30000;
        const auto q = uri.find('?');
        if (q == std::string::npos) {
            return timeout_ms;
        }
        const std::string query = uri.substr(q + 1);
        size_t pos = std::string::npos;
        if (query.rfind("timeout=", 0) == 0) {
            pos = 0;
        } else {
            const auto amp = query.find("&timeout=");
            if (amp != std::string::npos) {
                pos = amp + 1;
            }
        }
        if (pos == std::string::npos) {
            return timeout_ms;
        }
        timeout_ms = std::atoi(query.c_str() + pos + 8);
        if (timeout_ms <= 0) {
            timeout_ms = 30000;
        }
        if (timeout_ms > 300000) {
            timeout_ms = 300000;
        }
        return timeout_ms;
    }

    void reply_async_status(WFHttpTask* task, const std::string& id,
                            const typename AsyncTable::Snapshot& snap) {
        rapidjson::Document d;
        d.SetObject();
        auto& a = d.GetAllocator();
        d.AddMember("job_id", rapidjson::Value(id.c_str(), id.size(), a), a);
        d.AddMember("state", rapidjson::Value(async_state_str(snap.state), a), a);
        d.AddMember("elapsed_ms",
                    static_cast<int64_t>(async_now_ms() - snap.submitted_at_ms), a);
        if (!snap.error.empty()) {
            d.AddMember("error",
                        rapidjson::Value(snap.error.c_str(), snap.error.size(), a), a);
        }
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> w(buf);
        d.Accept(w);
        auto* resp = task->get_resp();
        resp->set_status_code("200");
        _metrics->inc_http_requests("GET", "200");
        resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
        resp->append_output_body(buf.GetString(), buf.GetSize());
    }

    void reply_async_error(WFHttpTask* task, int http_code, const std::string& msg) {
        const char* raw_method = task->get_req()->get_method();
        const std::string method = raw_method == nullptr ? "" : raw_method;
        _metrics->inc_http_requests(method, std::to_string(http_code));
        auto* resp = task->get_resp();
        resp->set_status_code(std::to_string(http_code).c_str());
        resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
        rapidjson::Document d;
        d.SetObject();
        auto& a = d.GetAllocator();
        d.AddMember("error", rapidjson::Value(msg.c_str(), msg.size(), a), a);
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> w(buf);
        d.Accept(w);
        resp->append_output_body(buf.GetString(), buf.GetSize());
    }

    void register_async_waiter(const std::string& job_id, const std::string& waiter_name) {
        std::lock_guard<std::mutex> lock(_waiters_mu);
        _waiters[job_id].push_back(waiter_name);
    }

    void unregister_async_waiter(const std::string& job_id, const std::string& waiter_name) {
        std::lock_guard<std::mutex> lock(_waiters_mu);
        auto it = _waiters.find(job_id);
        if (it == _waiters.end()) {
            return;
        }
        auto& names = it->second;
        names.erase(std::remove(names.begin(), names.end(), waiter_name), names.end());
        if (names.empty()) {
            _waiters.erase(it);
        }
    }

    void wake_async_waiters(const std::string& job_id) {
        std::vector<std::string> names;
        {
            std::lock_guard<std::mutex> lock(_waiters_mu);
            auto it = _waiters.find(job_id);
            if (it == _waiters.end()) {
                return;
            }
            names.swap(it->second);
            _waiters.erase(it);
        }
        for (const auto& name : names) {
            WFTaskFactory::count_by_name(name, 1);
        }
    }

    void handle_async_submit(WFHttpTask* task) {
        if (_table->queue_depth() >= _table->config().max_queue) {
            reply_async_error(task, 429, "async queue full (max " +
                                             std::to_string(_table->config().max_queue) +
                                             ")");
            return;
        }

        auto parsed = _parse(task->get_req());
        const std::string task_id = parsed.req_id.empty() ? generate_req_id() : parsed.req_id;
        if (!parsed.is_valid) {
            std::vector<jinq::common::ResponseError> errors;
            errors.reserve(parsed.violations.size());
            for (const auto& violation : parsed.violations) {
                errors.push_back({violation.pointer, violation.message});
            }
            _metrics->inc_http_requests("POST",
                                        std::to_string(http_status_of(parsed.status)));
            reply_unified_json(task->get_resp(),
                               unified_rejection(task_id, parsed.status, std::move(errors)));
            return;
        }
        if (item_count_exceeds(parsed.items.size(), _max_request_items)) {
            std::vector<jinq::common::ResponseError> limit_error;
            limit_error.push_back({"/images", "too many items in one request (max " +
                                                    std::to_string(_max_request_items) + ")"});
            _metrics->inc_http_requests("POST", "413");
            reply_unified_json(task->get_resp(),
                               unified_rejection(task_id, StatusCode::REQUEST_ITEM_LIMIT,
                                                 std::move(limit_error)));
            return;
        }

        InferenceTask task_req;
        task_req.task_id = task_id;
        task_req.items = std::move(parsed.items);
        task_req.params = std::move(parsed.params);
        task_req.options = parsed.options;
        if (_timeout_ms > 0) {
            task_req.deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(_timeout_ms);
        }

        const auto submitted = _table->submit(std::move(task_req));
        if (submitted.status == AsyncTable::SubmitStatus::QUEUE_FULL) {
            reply_async_error(task, 429, "async queue full (max " +
                                             std::to_string(_table->config().max_queue) +
                                             ")");
            return;
        }
        _metrics->inc_async_jobs("submitted");
        _metrics->set_async_queue_depth(
            static_cast<size_t>(std::max(0, _table->queue_depth())));

        const std::string job_id = submitted.job_id;
        _inflight.fetch_add(1, std::memory_order_acq_rel);
        _schedule(job_id);

        const std::string location = "/jobs/" + job_id;
        const std::string poll_url = location;
        const std::string result_url = location + "/result";
        auto* resp = task->get_resp();
        resp->set_status_code("202");
        _metrics->inc_http_requests("POST", "202");
        resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
        resp->add_header_pair("Location", location.c_str());
        rapidjson::Document d;
        d.SetObject();
        auto& a = d.GetAllocator();
        d.AddMember("job_id", rapidjson::Value(job_id.c_str(), job_id.size(), a), a);
        d.AddMember("state", "pending", a);
        d.AddMember("poll_url", rapidjson::Value(poll_url.c_str(), poll_url.size(), a), a);
        d.AddMember("result_url",
                    rapidjson::Value(result_url.c_str(), result_url.size(), a), a);
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> w(buf);
        d.Accept(w);
        resp->append_output_body(buf.GetString(), buf.GetSize());
    }

    void handle_async_status(WFHttpTask* task, const std::string& id) {
        auto snap = _table->snapshot(id);
        if (!snap.has_value()) {
            reply_async_error(task, 404, "job not found: " + id);
            return;
        }
        reply_async_status(task, id, *snap);
    }

    void handle_async_wait(WFHttpTask* task, const std::string& id, const std::string& uri) {
        auto initial = _table->snapshot(id);
        if (!initial.has_value()) {
            reply_async_error(task, 404, "job not found: " + id);
            return;
        }
        if (is_async_terminal(initial->state)) {
            reply_async_status(task, id, *initial);
            return;
        }
        const int timeout_ms = parse_wait_timeout_ms(uri);
        const std::string waiter_name =
            "mortred.jobwait." + id + "." +
            std::to_string(_wait_seq.fetch_add(1, std::memory_order_relaxed));
        register_async_waiter(id, waiter_name);
        _wait_inflight.fetch_add(1, std::memory_order_acq_rel);

        auto* counter = WFTaskFactory::create_counter_task(
            waiter_name, 1,
            [this, task, id, waiter_name](WFCounterTask*) {
                unregister_async_waiter(id, waiter_name);
                auto snap = _table->snapshot(id);
                if (!snap.has_value()) {
                    reply_async_error(task, 404, "job not found: " + id);
                } else {
                    reply_async_status(task, id, *snap);
                }
                _wait_inflight.fetch_sub(1, std::memory_order_acq_rel);
            });
        series_of(task)->push_back(counter);

        if (auto again = _table->snapshot(id);
            again.has_value() && is_async_terminal(again->state)) {
            WFTaskFactory::count_by_name(waiter_name, 1);
        }

        auto* timer = WFTaskFactory::create_timer_task(
            static_cast<time_t>(timeout_ms / 1000),
            static_cast<long>(timeout_ms % 1000) * 1000000L,
            [waiter_name](WFTimerTask*) { WFTaskFactory::count_by_name(waiter_name, 1); });
        timer->start();
    }

    void handle_async_result(WFHttpTask* task, const std::string& id) {
        auto outcome = _table->take_result(id);
        if (outcome.status == AsyncTable::ResultStatus::NOT_FOUND) {
            reply_async_error(task, 404, "job not found: " + id);
            return;
        }
        if (outcome.status != AsyncTable::ResultStatus::READY) {
            reply_async_error(task, 409,
                              "job not finished (state: " +
                                  std::string(async_state_str(outcome.state)) + ")");
            return;
        }
        auto unified = inference_result_to_unified(
            outcome.task_id, _model_name != nullptr ? *_model_name : std::string{},
            outcome.value,
            [this](rapidjson::Document::AllocatorType& allocator, rapidjson::Document& data,
                   const MODEL_OUTPUT& output, const OutputOptions& options) {
                _fill(allocator, data, output, options);
            });
        reply_unified_json(task->get_resp(), unified);
        _metrics->inc_http_requests("GET", "200");
    }

    AsyncTable* _table = nullptr;
    PrometheusMetrics* _metrics = nullptr;
    std::string* _model_name = nullptr;
    ParseFn _parse;
    FillFn _fill;
    ScheduleFn _schedule;
    bool _enabled = false;
    int _timeout_ms = 300000;
    size_t _max_request_items = 16;
    std::atomic<int> _inflight{0};
    std::atomic<int> _wait_inflight{0};
    std::atomic<uint64_t> _wait_seq{0};
    std::mutex _waiters_mu;
    std::unordered_map<std::string, std::vector<std::string>> _waiters;
};

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_ASYNC_ENDPOINTS_H
