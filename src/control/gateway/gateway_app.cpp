/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: main.cpp (mortred-gateway)
* Date: 26-8-22
************************************************/

// Data-plane reverse proxy: single external entry that routes each model
// by catalog id (/v1/models/{id}/…) or by the legacy server_uri to its
// loopback model server. Stateless by design; the supervisor owns its
// lifecycle (restart on exit, config reload = restart).

#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <workflow/HttpMessage.h>
#include <workflow/HttpUtil.h>
#include <workflow/WFHttpServer.h>
#include <workflow/WFTaskFactory.h>
#include <workflow/Workflow.h>

#include "common/auth_token.h"
#include "common/listen_policy.h"
#include "common/stage_timing.h"
#include "common/process_stop.h"
#include "control/api_key_manager.h"
#include "control/catalog.h"
#include "control/control_config.h"
#include "control/http_reply.h"
#include "control/project_root.h"
#include "control/trust_tokens.h"
#include "server/prometheus_metrics.h"

#include "control/gateway/gateway_app.h"

namespace {

using mortred::control::ApiKey;
using mortred::control::AuthReason;
using mortred::control::ServerEntry;

std::string header_value(const protocol::HttpRequest* req, const std::string& name) {
    protocol::HttpHeaderCursor cursor(req);
    protocol::HttpMessageHeader header;
    std::string target = name;
    std::transform(target.begin(), target.end(), target.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (cursor.next(&header)) {
        std::string h(static_cast<const char*>(header.name), header.name_len);
        std::transform(h.begin(), h.end(), h.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (h == target) {
            return std::string(static_cast<const char*>(header.value), header.value_len);
        }
    }
    return "";
}

std::string trim_copy(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(b, e - b);
}

bool key_may_infer(const std::shared_ptr<const ApiKey>& key) {
    return mortred::control::ApiKeyManager::has_scope(key, "inference") ||
           mortred::control::ApiKeyManager::has_scope(key, "admin");
}

using mortred::control::reply_json;
using mortred::control::reply_unified_error;

void reply_error(WFHttpTask* task, int http_status, const std::string& msg) {
    // fixed internal messages only: no client data is reflected into JSON; the
    // request id is echoed verbatim so gateway-local failures stay correlatable
    const std::string request_id = header_value(task->get_req(), "x-request-id");
    if (!request_id.empty()) {
        task->get_resp()->add_header_pair("X-Request-ID", request_id.c_str());
    }
    reply_unified_error(task, http_status, msg);
}

std::string uri_path(const char* uri) {
    const std::string s(uri == nullptr ? "" : uri);
    const auto q = s.find('?');
    return q == std::string::npos ? s : s.substr(0, q);
}

std::string uri_query(const char* uri) {
    const std::string s(uri == nullptr ? "" : uri);
    const auto q = s.find('?');
    return q == std::string::npos ? "" : s.substr(q);
}

std::string rewrite_jobs_public_url(const std::string& value, const std::string& id) {
    if (value.rfind("/jobs", 0) == 0) {
        return "/v1/models/" + id + value;
    }
    return value;
}

constexpr int k_stage_count = 5;
constexpr int k_bucket_count = 17;
constexpr const char* k_stage_names[k_stage_count] = {
    "gw_in", "route", "auth", "fwd", "upstream"};
constexpr double k_bucket_bounds[k_bucket_count] = {
    0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000};
constexpr const char* k_bucket_les[k_bucket_count] = {
    "0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10", "25", "50",
    "100", "250", "500", "1000", "2000", "5000", "10000"};

double elapsed_ms(std::chrono::steady_clock::time_point end,
                  std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string format_metric_double(double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    return buf;
}

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if ((n % 2) == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

/*** Top-level server_time_ms only. The copy is parsed so the bytes forwarded
 * to the client stay untouched. */
bool read_top_level_server_time_ms(const void* data, size_t size, double* out) {
    if (data == nullptr || size == 0 || out == nullptr) {
        return false;
    }
    const std::string copy(static_cast<const char*>(data), size);
    rapidjson::Document doc;
    doc.Parse(copy.data(), copy.size());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("server_time_ms") ||
        !doc["server_time_ms"].IsNumber()) {
        return false;
    }
    const double value = doc["server_time_ms"].GetDouble();
    if (!std::isfinite(value) || value < 0.0) {
        return false;
    }
    *out = value;
    return true;
}

std::string rewrite_job_url_fields(const char* data, size_t size, const std::string& id) {
    rapidjson::Document d;
    d.Parse(data, size);
    if (d.HasParseError() || !d.IsObject()) {
        return std::string(data, size);
    }
    bool changed = false;
    auto rewrite_field = [&](const char* key) {
        if (!d.HasMember(key) || !d[key].IsString()) {
            return;
        }
        const std::string v = d[key].GetString();
        const std::string nv = rewrite_jobs_public_url(v, id);
        if (nv != v) {
            d[key].SetString(nv.c_str(), static_cast<rapidjson::SizeType>(nv.size()),
                             d.GetAllocator());
            changed = true;
        }
    };
    rewrite_field("poll_url");
    rewrite_field("result_url");
    if (!changed) {
        return std::string(data, size);
    }
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    d.Accept(writer);
    return std::string(buf.GetString(), buf.GetSize());
}

}  // namespace

namespace mortred {
namespace control {

GatewayApp::~GatewayApp() {
    stop_listen();
}

/*** comma-separated list -> trimmed non-empty items (trusted_proxies config) */
static std::vector<std::string> split_csv(const std::string& raw) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t comma = raw.find(',', start);
        const size_t end = comma == std::string::npos ? raw.size() : comma;
        std::string item = raw.substr(start, end - start);
        const size_t b = item.find_first_not_of(" \t");
        const size_t e = item.find_last_not_of(" \t");
        if (b != std::string::npos) {
            out.push_back(item.substr(b, e - b + 1));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

void GatewayApp::load_cors_origins(const std::string& raw) {
    cors_origins_.clear();
    if (raw.empty()) {
        return;
    }
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t comma = raw.find(',', start);
        const size_t end = comma == std::string::npos ? raw.size() : comma;
        const std::string item = trim_copy(raw.substr(start, end - start));
        if (!item.empty()) {
            cors_origins_.push_back(item);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
}

bool GatewayApp::origin_allowed(const std::string& origin) const {
    if (origin.empty()) {
        return false;
    }
    for (const auto& allowed : cors_origins_) {
        if (origin == allowed) {
            return true;
        }
    }
    return false;
}

void GatewayApp::maybe_add_cors(WFHttpTask* task) const {
    const std::string origin = header_value(task->get_req(), "origin");
    if (!origin_allowed(origin)) {
        return;
    }
    auto* resp = task->get_resp();
    resp->add_header_pair("Access-Control-Allow-Origin", origin.c_str());
    resp->add_header_pair("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    resp->add_header_pair("Access-Control-Allow-Headers", "Authorization, Content-Type");
    resp->add_header_pair("Vary", "Origin");
}

bool GatewayApp::resolve_route(const std::string& path, ResolvedRoute* out) const {
    static const std::string kPrefix = "/v1/models/";
    if (path.rfind(kPrefix, 0) == 0) {
        const std::string rest = path.substr(kPrefix.size());
        const auto slash = rest.find('/');
        if (slash == std::string::npos || slash == 0) {
            return false;
        }
        const std::string id = rest.substr(0, slash);
        const std::string suffix = rest.substr(slash);
        // Exact catalog id. /v1/models/YOLOV8/infer and /v1/models/YOLOV8S/infer
        // do not alias yolov8s; there is no case fold and no redirect.
        const auto* entry = catalog_.find(id);
        if (entry == nullptr) {
            return false;
        }
        out->entry = entry;
        if (suffix == "/infer") {
            out->upstream_path = entry->uri;
            out->allowed_method = "POST";
            return true;
        }
        if (suffix == "/jobs") {
            out->upstream_path = "/jobs";
            out->allowed_method = "POST";
            out->rewrite_job_urls = true;
            return true;
        }
        if (suffix.rfind("/jobs/", 0) == 0) {
            out->upstream_path = suffix;
            out->allowed_method = "GET";
            out->rewrite_job_urls = true;
            out->append_query = true;
            return true;
        }
        return false;
    }
    const auto* entry = catalog_.find_by_uri(path);
    if (entry == nullptr) {
        return false;
    }
    out->entry = entry;
    out->upstream_path = entry->uri;
    out->allowed_method = "POST";
    return true;
}

void GatewayApp::forward_to_model(WFHttpTask* task, const ResolvedRoute& route,
                                  const std::string& method, const std::string& query,
                                  const std::shared_ptr<jinq::common::stage_timing::StageTrace>& stage_trace,
                                  const StageClocks& clocks) {
    struct ForwardClock {
        StageClocks clocks;
        std::chrono::steady_clock::time_point fwd{};
    };
    auto timing = std::make_shared<ForwardClock>();
    timing->clocks = clocks;

    std::string body;
    if (method == "POST") {
        body = protocol::HttpUtil::decode_chunked_body(task->get_req());
    }
    const std::string url =
        "http://127.0.0.1:" + std::to_string(route.entry->port) + route.upstream_path + query;
    const std::string model_id = route.entry->id;
    const bool rewrite_job_urls = route.rewrite_job_urls;
    const int send_timeout = cfg_.gateway.upstream_send_timeout_ms;
    const int recv_timeout = cfg_.gateway.upstream_recv_timeout_ms;
    const std::string internal_token = internal_token_;
    const auto t0 = std::chrono::steady_clock::now();

    auto* client = WFTaskFactory::create_http_task(
        url, 0, 0,
        [this, task, model_id, method, t0, rewrite_job_urls, stage_trace, timing](WFHttpTask* t) {
            auto* resp = task->get_resp();
            if (t->get_state() != WFT_STATE_SUCCESS) {
                const int code = t->get_error() == ECONNREFUSED ? 503 : 502;
                metrics_.inc_http_requests(method, std::to_string(code));
                metrics_.observe_http_duration_ms(
                    method, std::to_string(code),
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count());
                reply_error(task, code,
                            code == 503
                                ? "model server not running or still loading; "
                                  "TensorRT ids need mortredctl prepare"
                                : "upstream transport failure");
                if (stage_trace != nullptr) {
                    stage_trace->mark("upstream_fail");
                    std::fprintf(stderr, "%s\n", stage_trace->to_log_line("gateway").c_str());
                }
                return;
            }
            const auto t_upstream = std::chrono::steady_clock::now();
            if (stage_trace != nullptr) {
                stage_trace->mark("upstream");
            }
            const std::string status(t->get_resp()->get_status_code());
            metrics_.inc_http_requests(method, status);
            metrics_.observe_http_duration_ms(
                method, status,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count());
            resp->set_status_code(t->get_resp()->get_status_code());
            resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
            resp->add_header_pair("X-Mortred-Model", model_id.c_str());
            protocol::HttpHeaderCursor cursor(t->get_resp());
            protocol::HttpMessageHeader header;
            while (cursor.next(&header)) {
                std::string name(static_cast<const char*>(header.name), header.name_len);
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (name == "retry-after") {
                    const std::string value(static_cast<const char*>(header.value),
                                            header.value_len);
                    resp->add_header_pair("Retry-After", value.c_str());
                } else if (name == "location" && rewrite_job_urls) {
                    const std::string value(static_cast<const char*>(header.value),
                                            header.value_len);
                    const std::string public_loc = rewrite_jobs_public_url(value, model_id);
                    resp->add_header_pair("Location", public_loc.c_str());
                }
            }
            const void* data = nullptr;
            size_t size = 0;
            t->get_resp()->get_parsed_body(&data, &size);
            note_upstream_stage(timing->clocks, timing->fwd, t_upstream, rewrite_job_urls, status,
                                data, size);
            if (rewrite_job_urls && data != nullptr && size > 0) {
                const std::string rewritten = rewrite_job_url_fields(
                    static_cast<const char*>(data), size, model_id);
                resp->append_output_body(rewritten.data(), rewritten.size());
            } else {
                resp->append_output_body(data, size);
            }
            if (stage_trace != nullptr) {
                std::fprintf(stderr, "%s\n", stage_trace->to_log_line("gateway").c_str());
            }
        });
    client->get_req()->set_method(method.c_str());
    if (method == "POST") {
        client->get_req()->append_output_body(body.data(), body.size());
        // forward the client's encoding choice verbatim: the unified contract is
        // encoding-agnostic (JSON envelope today, raw image/* bodies in M6) and
        // the model server rejects what it does not support with a precise 415
        const std::string client_content_type = header_value(task->get_req(), "content-type");
        client->get_req()->add_header_pair(
            "Content-Type",
            client_content_type.empty() ? "application/json; charset=utf-8"
                                        : client_content_type.c_str());
    }
    const std::string client_accept = header_value(task->get_req(), "accept");
    if (!client_accept.empty()) {
        client->get_req()->add_header_pair("Accept", client_accept.c_str());
    }
    if (!internal_token.empty()) {
        const std::string auth = "Bearer " + internal_token;
        client->get_req()->add_header_pair("Authorization", auth.c_str());
    }
    // client-controlled headers with model-server semantics: raw-body params /
    // options and request correlation. Everything else stays dropped - the
    // default-deny forward list is the gateway's header-injection guard.
    static const char* const kForwardedClientHeaders[] = {
        "X-Mortred-Params", "X-Mortred-Options", "X-Request-ID"};
    for (const char* name : kForwardedClientHeaders) {
        const std::string value = header_value(task->get_req(), name);
        if (!value.empty()) {
            client->get_req()->add_header_pair(name, value.c_str());
        }
    }
    client->set_send_timeout(send_timeout);
    client->set_receive_timeout(recv_timeout);
    timing->fwd = std::chrono::steady_clock::now();
    if (stage_trace != nullptr) {
        stage_trace->mark("fwd");
    }
    series_of(task)->push_back(client);
}

void GatewayApp::note_upstream_stage(const StageClocks& clocks,
                                     std::chrono::steady_clock::time_point t_fwd,
                                     std::chrono::steady_clock::time_point t_upstream,
                                     bool job_route, const std::string& status,
                                     const void* data, size_t size) {
    if (job_route || status != "200" || data == nullptr || size == 0) {
        return;
    }
    double server_time_ms = 0.0;
    if (!read_top_level_server_time_ms(data, size, &server_time_ms)) {
        return;
    }
    const double stage_ms[k_stage_count] = {
        elapsed_ms(clocks.gw_in, clocks.enter),
        elapsed_ms(clocks.route, clocks.gw_in),
        elapsed_ms(clocks.auth, clocks.route),
        elapsed_ms(t_fwd, clocks.auth),
        elapsed_ms(t_upstream, t_fwd),
    };
    record_stage_sample(stage_ms, server_time_ms);
}

void GatewayApp::record_stage_sample(const double stage_ms[5], double server_time_ms) {
    static_assert(k_stage_count == 5, "five gateway stages");
    static_assert(k_stage_bucket_count == k_bucket_count, "stage histogram buckets");
    std::lock_guard<std::mutex> lock(stage_mu_);
    if (stage_samples_.size() == k_stage_sample_cap) {
        stage_samples_.pop_front();
    }
    StageSample sample;
    sample.server_time_ms = server_time_ms;
    for (int i = 0; i < k_stage_count; ++i) {
        sample.ms[i] = stage_ms[i];
        stage_sum_[i] += stage_ms[i];
        for (int bucket = 0; bucket < k_stage_bucket_count; ++bucket) {
            if (stage_ms[i] <= k_bucket_bounds[bucket]) {
                ++stage_bucket_[i][bucket];
            }
        }
    }
    stage_samples_.push_back(sample);
    ++stage_observed_;
}

std::string GatewayApp::render_stage_metrics() const {
    std::lock_guard<std::mutex> lock(stage_mu_);
    if (stage_observed_ == 0) {
        return "";
    }
    std::ostringstream ss;
    ss << "# HELP mortred_gateway_stage_duration_ms Gateway stage duration in milliseconds\n";
    ss << "# TYPE mortred_gateway_stage_duration_ms histogram\n";
    for (int i = 0; i < k_stage_count; ++i) {
        for (int bucket = 0; bucket < k_stage_bucket_count; ++bucket) {
            ss << "mortred_gateway_stage_duration_ms_bucket{model=\"gateway\",stage=\""
               << k_stage_names[i] << "\",le=\"" << k_bucket_les[bucket] << "\"} "
               << stage_bucket_[i][bucket] << "\n";
        }
        ss << "mortred_gateway_stage_duration_ms_bucket{model=\"gateway\",stage=\""
           << k_stage_names[i] << "\",le=\"+Inf\"} " << stage_observed_ << "\n";
        ss << "mortred_gateway_stage_duration_ms_sum{model=\"gateway\",stage=\""
           << k_stage_names[i] << "\"} " << format_metric_double(stage_sum_[i]) << "\n";
        ss << "mortred_gateway_stage_duration_ms_count{model=\"gateway\",stage=\""
           << k_stage_names[i] << "\"} " << stage_observed_ << "\n";
    }
    if (!stage_samples_.empty()) {
        double stage_p50[k_stage_count];
        for (int i = 0; i < k_stage_count; ++i) {
            std::vector<double> values;
            values.reserve(stage_samples_.size());
            for (const auto& sample : stage_samples_) {
                values.push_back(sample.ms[i]);
            }
            stage_p50[i] = median_of(std::move(values));
        }
        std::vector<double> server_times;
        server_times.reserve(stage_samples_.size());
        for (const auto& sample : stage_samples_) {
            server_times.push_back(sample.server_time_ms);
        }
        const double server_p50 = median_of(std::move(server_times));
        double gateway_p50 = 0.0;
        for (int i = 0; i < k_stage_count; ++i) {
            gateway_p50 += stage_p50[i];
        }
        if (gateway_p50 > 0.5 * server_p50) {
            int winner = 0;
            for (int i = 1; i < k_stage_count; ++i) {
                // Equal p50 keeps the later name: gw_in, route, auth, fwd, upstream.
                if (stage_p50[i] >= stage_p50[winner]) {
                    winner = i;
                }
            }
            ss << "# HELP mortred_gateway_stage_dominant Time is mainly spent in this gateway stage\n";
            ss << "# TYPE mortred_gateway_stage_dominant gauge\n";
            ss << "mortred_gateway_stage_dominant{model=\"gateway\",stage=\""
               << k_stage_names[winner] << "\"} 1\n";
        }
    }
    return ss.str();
}

/*** L1 per-source metering (PR-3b). Runs BEFORE route resolution and auth.
 * init turns counting on only after [gateway.rate_limit] and
 * MORTRED_RATE_LIMIT (off / shadow / enforce). enabled and shadow both
 * false: return before admit, write no 429, and leave no per-source
 * budget for this request. rate_per_sec and burst stay in config and
 * apply only while counting. Ordering inside process():
 *   OPTIONS -> /healthz -> /metrics -> [THIS] -> route -> auth(L2) -> forward
 * 429 carries Retry-After (exact GCRA budget, rounded up to whole seconds)
 * and the IETF draft RateLimit-Limit/Remaining/Reset triple. Enforce uses
 * the configured rate_per_sec and burst. Shadow mode counts but never
 * rejects. proceed is false only when enforce already wrote the 429.
 * An allowed admit is not released here: process() does that on the
 * pre-forward 401 path, for both enforce and shadow. */
GatewayApp::IpRateGate GatewayApp::check_ip_rate_limit(WFHttpTask* task, const std::string& method) {
    IpRateGate gate;
    if (!rate_limit_enabled_ || ip_limiter_ == nullptr || method == "OPTIONS") {
        return gate;
    }
    // peer address from the TCP endpoint (same shape as the model-server
    // peer_ip_of; workflow gives sockaddr storage directly)
    struct sockaddr_storage peer_addr;
    socklen_t addr_len = sizeof(peer_addr);
    char ip_buf[INET6_ADDRSTRLEN] = {0};
    if (task->get_peer_addr(reinterpret_cast<struct sockaddr*>(&peer_addr), &addr_len) == 0) {
        if (peer_addr.ss_family == AF_INET) {
            const auto* v4 = reinterpret_cast<const struct sockaddr_in*>(&peer_addr);
            ::inet_ntop(AF_INET, &v4->sin_addr, ip_buf, sizeof(ip_buf));
        } else if (peer_addr.ss_family == AF_INET6) {
            const auto* v6 = reinterpret_cast<const struct sockaddr_in6*>(&peer_addr);
            ::inet_ntop(AF_INET6, &v6->sin6_addr, ip_buf, sizeof(ip_buf));
        }
    }
    const bool used_header = false;
    gate.subject = mortred::control::ratelimit::derive_subject(
        ip_buf, trusted_proxies_,
        header_value(task->get_req(), "forwarded"),
        header_value(task->get_req(), "x-forwarded-for"));
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    const auto decision = ip_limiter_->admit(gate.subject, now_ms);
    if (decision.allowed) {
        gate.admitted = true;
        return gate;
    }
    const int retry_s = (decision.retry_after_ms + 999) / 1000;
    if (rate_limit_shadow_) {
        metrics_.inc_http_requests(method, "shadow429");
        return gate;
    }
    metrics_.inc_http_requests(method, "429");
    auto* resp = task->get_resp();
    resp->add_header_pair("Retry-After", std::to_string(retry_s).c_str());
    // IETF draft rate-limit header fields
    resp->add_header_pair("RateLimit-Limit", std::to_string(rate_limit_rate_).c_str());
    resp->add_header_pair("RateLimit-Remaining", "0");
    resp->add_header_pair("RateLimit-Reset", std::to_string(retry_s).c_str());
    reply_unified_error(task, 429, jinq::common::StatusCode::RATE_LIMITED,
                        "per-source rate limit exceeded");
    gate.proceed = false;
    return gate;
}

void GatewayApp::process(WFHttpTask* task) {
    const auto t_enter = std::chrono::steady_clock::now();
    const std::string path = uri_path(task->get_req()->get_request_uri());
    // workflow leaves method null when the request line is malformed; uri_path
    // already folds a null uri into "", do the same for the method
    const char* raw_method = task->get_req()->get_method();
    const std::string method = raw_method == nullptr ? "" : raw_method;

    maybe_add_cors(task);
    if (method == "OPTIONS") {
        task->get_resp()->set_status_code("204");
        return;
    }

    if (path == "/healthz") {
        task->get_resp()->set_status_code("200");
        task->get_resp()->add_header_pair("Content-Type", "text/plain; charset=utf-8");
        task->get_resp()->append_output_body("ok\n", 3);
        return;
    }
    if (path == "/metrics") {
        const std::string auth_header = header_value(task->get_req(), "authorization");
        if (!jinq::common::is_bearer_authorized(auth_header, metrics_token_)) {
            metrics_.inc_http_requests(method, "401");
            task->get_resp()->add_header_pair("WWW-Authenticate",
                                             "Bearer realm=\"Mortred\"");
            reply_error(task, 401, "unauthorized");
            return;
        }
        auto* resp = task->get_resp();
        resp->set_status_code("200");
        resp->add_header_pair("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        const auto body = metrics_.render();
        const auto stages = render_stage_metrics();
        resp->append_output_body(body.data(), body.size());
        if (!stages.empty()) {
            resp->append_output_body(stages.data(), stages.size());
        }
        return;
    }

    auto stage_trace = jinq::common::stage_timing::enabled()
                           ? std::make_shared<jinq::common::stage_timing::StageTrace>()
                           : nullptr;
    StageClocks clocks;
    clocks.enter = t_enter;
    clocks.gw_in = std::chrono::steady_clock::now();
    if (stage_trace != nullptr) {
        stage_trace->mark("gw_in");
    }

    const IpRateGate ip_gate = check_ip_rate_limit(task, method);
    if (!ip_gate.proceed) {
        return;
    }

    ResolvedRoute route;
    if (!resolve_route(path, &route)) {
        metrics_.inc_http_requests(method, "404");
        reply_error(task, 404, "no model route for this path");
        return;
    }
    clocks.route = std::chrono::steady_clock::now();
    if (stage_trace != nullptr) {
        stage_trace->mark("route");
    }
    // external auth is enforced here, once, for every model endpoint
    const std::string auth_header = header_value(task->get_req(), "authorization");
    bool authorized = false;
    std::string key_name;

    // try multi-key auth first (P0-4)
    if (api_keys_.key_count() > 0) {
        // shared_ptr ownership: safe across a concurrent reload() that swaps
        // the whole key set (P0-2). The per-key window increments only after
        // key_may_infer is already true, so a scope that cannot infer keeps
        // the existing 401 and does not take a rate_window_count slot.
        const auto auth = api_keys_.identify(auth_header);
        if (key_may_infer(auth.key)) {
            const auto limited = api_keys_.consume_rate_limit(auth.key);
            if (limited.reason == AuthReason::RATE_LIMITED) {
                // a VALID key that is merely throttled: 429 + Retry-After, never
                // 401, and never a fallthrough to the static-token path (the
                // client presented working credentials)
                metrics_.inc_http_requests(method, "429");
                const int retry_s = (limited.retry_after_ms + 999) / 1000;
                task->get_resp()->add_header_pair("Retry-After",
                                                  std::to_string(retry_s).c_str());
                reply_unified_error(task, 429, jinq::common::StatusCode::RATE_LIMITED,
                                    "api key rate limit exceeded");
                return;
            }
            authorized = true;
            key_name = auth.key->name;
        }
    }

    // fallback to single static token (legacy compatibility). The empty-token
    // "auth disabled" semantics of is_bearer_authorized belongs ONLY to the
    // nothing-configured case; once API keys are configured, an
    // unauthenticated request must stay denied instead of falling through to
    // the empty static token (fail-open).
    if (!authorized && !auth_token_.empty() &&
        jinq::common::is_bearer_authorized(auth_header, auth_token_)) {
        authorized = true;
        key_name = "legacy";
    }
    if (!authorized && !admin_token_.empty() &&
        jinq::common::is_bearer_authorized(auth_header, admin_token_)) {
        authorized = true;
        key_name = "admin";
    }

    if (!authorized) {
        // Allowed admit, 401 before forward: give back one emission interval.
        // Enforce and shadow share this path. 404/405/502/503 and key 429 do not.
        if (ip_gate.admitted && ip_limiter_ != nullptr) {
            ip_limiter_->release(ip_gate.subject);
        }
        metrics_.inc_http_requests(method, "401");
        task->get_resp()->add_header_pair("WWW-Authenticate", "Bearer realm=\"Mortred\"");
        reply_error(task, 401, "unauthorized");
        return;
    }
    // tag the request with the key name for upstream logging
    if (!key_name.empty()) {
        task->get_resp()->add_header_pair("X-Mortred-Key", key_name.c_str());
    }
    clocks.auth = std::chrono::steady_clock::now();
    if (stage_trace != nullptr) {
        stage_trace->mark("auth");
    }
    if (method != route.allowed_method) {
        metrics_.inc_http_requests(method, "405");
        task->get_resp()->add_header_pair("Allow", route.allowed_method.c_str());
        reply_error(task, 405, "method not allowed");
        return;
    }
    const std::string query =
        route.append_query ? uri_query(task->get_req()->get_request_uri()) : "";
    forward_to_model(task, route, method, query, stage_trace, clocks);
}

bool GatewayApp::init(const GatewayInitOptions& options) {
    std::string config_path = options.config_path;
    if (config_path.empty()) {
        config_path =
            (std::filesystem::path(options.project_root) / "conf" / "mortred.toml").string();
    }
    std::string cfg_err;
    if (!ControlConfig::load(config_path, &cfg_, &cfg_err)) {
        std::fprintf(stderr, "mortred-gateway: invalid control config: %s\n", cfg_err.c_str());
        return false;
    }
    if (!options.host.empty()) {
        cfg_.gateway.host = options.host;
    }
    if (options.port > 0 && options.port <= 65535) {
        cfg_.gateway.port = options.port;
    }
    auth_token_ = options.auth_token;
    admin_token_ = options.admin_token;
    metrics_token_ = options.metrics_token;
    internal_token_ = options.internal_token;
    load_cors_origins(options.cors_origins);

    // L1 per-IP limiter: construct AFTER config load so [gateway.rate_limit]
    // from mortred.toml applies; MORTRED_RATE_LIMIT env (values: off|shadow|
    // enforce) overrides for docker/compose without editing the toml
    if (const char* env = std::getenv("MORTRED_RATE_LIMIT"); env != nullptr &&
        *env != '\0') {
        const std::string mode(env);
        cfg_.gateway.rate_limit.enabled = (mode == "enforce");
        cfg_.gateway.rate_limit.shadow = (mode == "shadow");
    }
    const auto& rl = cfg_.gateway.rate_limit;
    rate_limit_enabled_ = rl.enabled || rl.shadow;
    rate_limit_shadow_ = rl.shadow;
    rate_limit_rate_ = rl.rate_per_sec;
    trusted_proxies_ = mortred::control::ratelimit::TrustedProxies::parse(
        split_csv(rl.trusted_proxies));
    // Count only when enabled or shadow is on, including after
    // MORTRED_RATE_LIMIT=off has cleared both flags. rate_per_sec and burst
    // stay on the config and are not applied while the limiter is unset.
// P0-2: public/non-loopback listen must not run with rate limiting fully off
// (neither enforce nor shadow). Operators who truly need that set
// MORTRED_RATE_LIMIT_ALLOW_DISABLE=1.
if (!rate_limit_enabled_) {
    const bool public_listen = !jinq::common::is_loopback_host(cfg_.gateway.host);
    const char* allow = std::getenv("MORTRED_RATE_LIMIT_ALLOW_DISABLE");
    const bool allow_disable = allow != nullptr && allow[0] != '\0'
        && (std::string(allow) == "1" || std::string(allow) == "true");
    if (public_listen && !allow_disable) {
        std::fprintf(stderr,
                     "mortred-gateway: refusing to start: gateway.rate_limit is off while "
                     "listening on non-loopback %s (set enabled=true, or MORTRED_RATE_LIMIT="
                     "enforce|shadow, or MORTRED_RATE_LIMIT_ALLOW_DISABLE=1)\n",
                     cfg_.gateway.host.c_str());
        return false;
    }
}

    if (rate_limit_enabled_) {
        ip_limiter_ = std::make_unique<mortred::control::ratelimit::ShardedIpLimiter>(
            mortred::control::ratelimit::GcraPolicy{
                static_cast<uint32_t>(rl.rate_per_sec), static_cast<uint32_t>(rl.burst)},
            static_cast<size_t>(rl.max_tracked));
        std::fprintf(stderr,
                     "mortred-gateway: per-IP rate limiting %s: %d/s burst %d, %zu trusted "
                     "proxies\n",
                     rate_limit_shadow_ ? "SHADOW (count, never reject)" : "ENFORCED",
                     rl.rate_per_sec, rl.burst, trusted_proxies_.entries.size());
    }

    // A loaded key file does not replace a missing inference or management token.
    bool api_keys_parse_failed = false;
    bool api_keys_empty_file = false;
    {
        const std::string api_keys_path =
            (std::filesystem::path(options.project_root) / "conf" / "api_keys.toml").string();
        if (std::filesystem::exists(api_keys_path)) {
            if (!api_keys_.load(api_keys_path)) {
                api_keys_parse_failed = true;
                std::fprintf(stderr, "mortred-gateway: ERROR: failed to parse %s\n",
                             api_keys_path.c_str());
            } else if (api_keys_.key_count() == 0) {
                api_keys_empty_file = true;
                std::fprintf(stderr,
                             "mortred-gateway: ERROR: %s parsed but contains no keys "
                             "(empty key file is not auth)\n",
                             api_keys_path.c_str());
            } else {
                std::fprintf(stderr, "mortred-gateway: loaded %zu API keys from %s\n",
                             api_keys_.key_count(), api_keys_path.c_str());
            }
        }
    }

    if (!jinq::common::listen_host_permitted(cfg_.gateway.host)) {
        std::fprintf(stderr,
                     "mortred-gateway: refusing to listen on %s (MORTRED_EXPOSE=%s). "
                     "Bind 127.0.0.1 and terminate TLS at Nginx (deploy/nginx). "
                     "Containers: MORTRED_EXPOSE=docker. Metal wildcard: MORTRED_EXPOSE=unsafe "
                     "(plaintext; doctor --strict fails)\n",
                     cfg_.gateway.host.c_str(),
                     jinq::common::mortred_expose_mode().c_str());
        return false;
    }
    const auto listen = tokens_usable_for_listen(metrics_token_, auth_token_, admin_token_);
    if (!listen.ok) {
        if (std::strcmp(listen.reason, kScrapeTokenMissing) == 0) {
            std::fprintf(stderr,
                         "mortred-gateway: refusing to start without MORTRED_METRICS_TOKEN "
                         "(GET /metrics is never public, including loopback; set a scrape Bearer "
                         "distinct from the inference and management tokens). "
                         "Generate with: mortredctl init-trust\n");
        } else if (std::strcmp(listen.reason, kScrapeTokenCollision) == 0) {
            std::fprintf(stderr,
                         "mortred-gateway: refusing to start: MORTRED_METRICS_TOKEN matches an "
                         "inference or management token; Prometheus would then hold that privilege\n");
        } else {
            std::fprintf(stderr, "mortred-gateway: refusing to start: %s\n", listen.reason);
        }
        return false;
    }
    if (api_keys_parse_failed) {
        std::fprintf(stderr,
                     "mortred-gateway: WARNING: conf/api_keys.toml failed to parse; continuing "
                     "with static-token auth only\n");
    } else if (api_keys_empty_file) {
        std::fprintf(stderr,
                     "mortred-gateway: WARNING: conf/api_keys.toml has no keys; continuing "
                     "with static-token auth only\n");
    }

    std::string runtime_profile = options.profile;
    if (runtime_profile.empty()) {
        const char* profile_env = std::getenv("MORTRED_PROFILE");
        runtime_profile =
            (profile_env != nullptr && std::string(profile_env) == "cpu") ? "cpu" : "gpu";
    }
    std::string catalog_err;
    if (!catalog_.init(options.project_root, &catalog_err, runtime_profile)) {
        std::fprintf(stderr, "mortred-gateway: catalog init failed (profile=%s): %s\n",
                     runtime_profile.c_str(), catalog_err.c_str());
        return false;
    }
    metrics_.set_model("gateway");
    return true;
}

bool GatewayApp::listen() {
    WFServerParams params = SERVER_PARAMS_DEFAULT;
    params.max_connections = cfg_.gateway.max_connections;
    params.request_size_limit =
        static_cast<size_t>(cfg_.gateway.request_size_limit_mb) * 1024 * 1024;
    server_ = std::make_unique<WFHttpServer>(
        &params, [this](WFHttpTask* task) { process(task); });
    if (server_->start(cfg_.gateway.host.c_str(),
                       static_cast<unsigned short>(cfg_.gateway.port)) != 0) {
        std::fprintf(stderr, "mortred-gateway: cannot listen on %s:%d\n",
                     cfg_.gateway.host.c_str(), cfg_.gateway.port);
        server_.reset();
        return false;
    }
    return true;
}

void GatewayApp::stop_listen() {
    if (server_ != nullptr) {
        server_->stop();
        server_.reset();
    }
}

int GatewayApp::run(int argc, char** argv) {
    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        std::fprintf(stderr, "usage: mortred-gateway.out (config via env / conf/mortred.toml)\n");
        return 0;
    }

    GatewayInitOptions options;
    options.project_root = resolve_project_root();
    if (const char* env = std::getenv("MORTRED_CONTROL_CONFIG");
        env != nullptr && *env != '\0') {
        options.config_path = env;
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_HOST"); env != nullptr && *env != '\0') {
        options.host = env;
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_PORT"); env != nullptr && *env != '\0') {
        const int port = std::atoi(env);
        if (port > 0 && port <= 65535) {
            options.port = port;
        }
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_AUTH_TOKEN");
        env != nullptr && *env != '\0') {
        options.auth_token = env;
    }
    if (const char* env = std::getenv("MORTRED_API_TOKEN"); env != nullptr && *env != '\0') {
        options.admin_token = env;
    }
    if (const char* env = std::getenv("MORTRED_METRICS_TOKEN"); env != nullptr && *env != '\0') {
        options.metrics_token = env;
    }
    if (const char* env = std::getenv("MORTRED_INTERNAL_TOKEN"); env != nullptr && *env != '\0') {
        options.internal_token = env;
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_CORS_ORIGINS");
        env != nullptr && *env != '\0') {
        options.cors_origins = env;
    }
    if (!init(options)) {
        return 1;
    }

    jinq::common::ProcessStop process_stop;
    process_stop.arm();

    if (!listen()) {
        return 1;
    }
    const char* auth_mode =
        api_keys_.key_count() > 0 ? "api-keys auth" : "static-token auth";
    std::fprintf(stderr,
                 "mortred-gateway listening on http://%s:%d (routes: %zu) [%s, metrics scrape "
                 "token, expose=%s]\n",
                 cfg_.gateway.host.c_str(), cfg_.gateway.port, catalog_.entries().size(),
                 auth_mode, jinq::common::mortred_expose_mode().c_str());

    process_stop.wait();
    stop_listen();
    return 0;
}

int run_gateway(int argc, char** argv) {
    return GatewayApp().run(argc, argv);
}

}  // namespace control
}  // namespace mortred
