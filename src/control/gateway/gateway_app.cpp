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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
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
#include "common/process_stop.h"
#include "control/api_key_manager.h"
#include "control/catalog.h"
#include "control/control_config.h"
#include "control/http_reply.h"
#include "server/prometheus_metrics.h"

#include "control/gateway/gateway_app.h"

namespace {

using mortred::control::Catalog;
using mortred::control::ControlConfig;

Catalog g_catalog;
ControlConfig g_cfg;
std::string g_auth_token;       // external bearer token ("" = loopback mode)
std::string g_admin_token;      // MORTRED_API_TOKEN; UI / mortredctl smoke
std::string g_metrics_token;    // scrape Bearer; required on every listen, including loopback
mortred::control::ApiKeyManager g_api_keys;  // multi-key auth (P0-4)
std::string g_internal_token;   // shared with model servers via supervisor env
std::vector<std::string> g_cors_origins;  // UI origins allowed to call infer
jinq::server::PrometheusMetrics g_metrics;

std::string resolve_project_root() {
    if (const char* env = std::getenv("MORTRED_PROJECT_ROOT"); env != nullptr && *env != '\0') {
        return env;
    }
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    buf[n] = '\0';
    std::filesystem::path p(buf);
    auto dir = p.parent_path();
    for (int i = 0; i < 12 && !dir.empty(); ++i) {
        std::error_code ec;
        const bool has_bin = std::filesystem::exists(dir / "_bin", ec) ||
                             std::filesystem::exists(dir / "bin", ec);
        ec.clear();
        const bool has_deps = std::filesystem::exists(dir / "3rd_party", ec) ||
                              std::filesystem::exists(dir / "lib", ec);
        if (has_bin && has_deps) {
            return dir.string();
        }
        dir = dir.parent_path();
    }
    return ".";
}

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

void load_cors_origins(const char* env) {
    g_cors_origins.clear();
    if (env == nullptr || *env == '\0') {
        return;
    }
    const std::string raw(env);
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t comma = raw.find(',', start);
        const size_t end = comma == std::string::npos ? raw.size() : comma;
        const std::string item = trim_copy(raw.substr(start, end - start));
        if (!item.empty()) {
            g_cors_origins.push_back(item);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
}

bool origin_allowed(const std::string& origin) {
    if (origin.empty()) {
        return false;
    }
    for (const auto& allowed : g_cors_origins) {
        if (origin == allowed) {
            return true;
        }
    }
    return false;
}

void maybe_add_cors(WFHttpTask* task) {
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

bool key_may_infer(const std::shared_ptr<const mortred::control::ApiKey>& key) {
    return mortred::control::ApiKeyManager::has_scope(key, "inference") ||
           mortred::control::ApiKeyManager::has_scope(key, "admin");
}

using mortred::control::reply_json;
using mortred::control::reply_unified_error;

void reply_error(WFHttpTask* task, int http_status, const std::string& msg) {
    // fixed internal messages only: no client data is reflected into JSON
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

struct ResolvedRoute {
    const mortred::control::ServerEntry* entry = nullptr;
    std::string upstream_path;
    std::string allowed_method;
    bool rewrite_job_urls = false;
    bool append_query = false;
};

bool resolve_route(const std::string& path, ResolvedRoute* out) {
    static const std::string kPrefix = "/v1/models/";
    if (path.rfind(kPrefix, 0) == 0) {
        const std::string rest = path.substr(kPrefix.size());
        const auto slash = rest.find('/');
        if (slash == std::string::npos || slash == 0) {
            return false;
        }
        const std::string id = rest.substr(0, slash);
        const std::string suffix = rest.substr(slash);
        const auto* entry = g_catalog.find(id);
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
    const auto* entry = g_catalog.find_by_uri(path);
    if (entry == nullptr) {
        return false;
    }
    out->entry = entry;
    out->upstream_path = entry->uri;
    out->allowed_method = "POST";
    return true;
}

void forward_to_model(WFHttpTask* task, const ResolvedRoute& route, const std::string& method,
                      const std::string& query) {
    std::string body;
    if (method == "POST") {
        body = protocol::HttpUtil::decode_chunked_body(task->get_req());
    }
    const std::string url =
        "http://127.0.0.1:" + std::to_string(route.entry->port) + route.upstream_path + query;
    const std::string model_id = route.entry->id;
    const bool rewrite_job_urls = route.rewrite_job_urls;
    const int send_timeout = g_cfg.gateway.upstream_send_timeout_ms;
    const int recv_timeout = g_cfg.gateway.upstream_recv_timeout_ms;
    const std::string internal_token = g_internal_token;
    const auto t0 = std::chrono::steady_clock::now();

    auto* client = WFTaskFactory::create_http_task(
        url, 0, 0, [task, model_id, method, t0, rewrite_job_urls](WFHttpTask* t) {
            auto* resp = task->get_resp();
            if (t->get_state() != WFT_STATE_SUCCESS) {
                const int code = t->get_error() == ECONNREFUSED ? 503 : 502;
                g_metrics.inc_http_requests(method, std::to_string(code));
                g_metrics.observe_http_duration_ms(
                    method, std::to_string(code),
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count());
                reply_error(task, code,
                            code == 503
                                ? "model server not running or still loading; "
                                  "TensorRT ids need mortredctl prepare"
                                : "upstream transport failure");
                return;
            }
            const std::string status(t->get_resp()->get_status_code());
            g_metrics.inc_http_requests(method, status);
            g_metrics.observe_http_duration_ms(
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
            if (rewrite_job_urls && data != nullptr && size > 0) {
                const std::string rewritten = rewrite_job_url_fields(
                    static_cast<const char*>(data), size, model_id);
                resp->append_output_body(rewritten.data(), rewritten.size());
            } else {
                resp->append_output_body(data, size);
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
    client->set_send_timeout(send_timeout);
    client->set_receive_timeout(recv_timeout);
    series_of(task)->push_back(client);
}

void process(WFHttpTask* task) {
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
        if (!jinq::common::is_bearer_authorized(auth_header, g_metrics_token)) {
            g_metrics.inc_http_requests(method, "401");
            task->get_resp()->add_header_pair("WWW-Authenticate",
                                             "Bearer realm=\"Mortred\"");
            reply_error(task, 401, "unauthorized");
            return;
        }
        auto* resp = task->get_resp();
        resp->set_status_code("200");
        resp->add_header_pair("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        const auto body = g_metrics.render();
        resp->append_output_body(body.data(), body.size());
        return;
    }

    ResolvedRoute route;
    if (!resolve_route(path, &route)) {
        g_metrics.inc_http_requests(method, "404");
        reply_error(task, 404, "no model route for this path");
        return;
    }
    // external auth is enforced here, once, for every model endpoint
    const std::string auth_header = header_value(task->get_req(), "authorization");
    bool authorized = false;
    std::string key_name;

    // try multi-key auth first (P0-4)
    if (g_api_keys.key_count() > 0) {
        // shared_ptr ownership: safe across a concurrent reload() that swaps
        // the whole key set (P0-2)
        const auto key = g_api_keys.authenticate(auth_header);
        if (key_may_infer(key)) {
            authorized = true;
            key_name = key->name;
        }
    }

    // fallback to single static token (legacy compatibility). The empty-token
    // "auth disabled" semantics of is_bearer_authorized belongs ONLY to the
    // nothing-configured case; once API keys are configured, an
    // unauthenticated request must stay denied instead of falling through to
    // the empty static token (fail-open).
    if (!authorized && !g_auth_token.empty() &&
        jinq::common::is_bearer_authorized(auth_header, g_auth_token)) {
        authorized = true;
        key_name = "legacy";
    }
    if (!authorized && !g_admin_token.empty() &&
        jinq::common::is_bearer_authorized(auth_header, g_admin_token)) {
        authorized = true;
        key_name = "admin";
    }

    if (!authorized) {
        g_metrics.inc_http_requests(method, "401");
        task->get_resp()->add_header_pair("WWW-Authenticate", "Bearer realm=\"Mortred\"");
        reply_error(task, 401, "unauthorized");
        return;
    }
    // tag the request with the key name for upstream logging
    if (!key_name.empty()) {
        task->get_resp()->add_header_pair("X-Mortred-Key", key_name.c_str());
    }
    if (method != route.allowed_method) {
        g_metrics.inc_http_requests(method, "405");
        task->get_resp()->add_header_pair("Allow", route.allowed_method.c_str());
        reply_error(task, 405, "method not allowed");
        return;
    }
    const std::string query = route.append_query ? uri_query(task->get_req()->get_request_uri()) : "";
    forward_to_model(task, route, method, query);
}

}  // namespace

namespace mortred {
namespace control {

int run_gateway(int argc, char** argv) {
    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        std::fprintf(stderr, "usage: mortred-gateway.out (config via env / conf/mortred.toml)\n");
        return 0;
    }

    const std::string root = resolve_project_root();
    std::string config_path;
    if (const char* env = std::getenv("MORTRED_CONTROL_CONFIG");
        env != nullptr && *env != '\0') {
        config_path = env;
    } else {
        config_path = (std::filesystem::path(root) / "conf" / "mortred.toml").string();
    }
    std::string cfg_err;
    if (!ControlConfig::load(config_path, &g_cfg, &cfg_err)) {
        std::fprintf(stderr, "mortred-gateway: invalid control config: %s\n", cfg_err.c_str());
        return 1;
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_HOST"); env != nullptr && *env != '\0') {
        g_cfg.gateway.host = env;
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_PORT"); env != nullptr && *env != '\0') {
        const int port = std::atoi(env);
        if (port > 0 && port <= 65535) {
            g_cfg.gateway.port = port;
        }
    }
    if (const char* env = std::getenv("MORTRED_GATEWAY_AUTH_TOKEN");
        env != nullptr && *env != '\0') {
        g_auth_token = env;
    }
    if (const char* env = std::getenv("MORTRED_API_TOKEN"); env != nullptr && *env != '\0') {
        g_admin_token = env;
    }
    if (const char* env = std::getenv("MORTRED_METRICS_TOKEN"); env != nullptr && *env != '\0') {
        g_metrics_token = env;
    }
    load_cors_origins(std::getenv("MORTRED_GATEWAY_CORS_ORIGINS"));
    if (const char* env = std::getenv("MORTRED_INTERNAL_TOKEN"); env != nullptr && *env != '\0') {
        g_internal_token = env;
    }

    // fail-closed: every listen needs inference/management auth. Broken or
    // empty api_keys.toml is fatal when no static token can take over.
    bool api_keys_loaded = false;
    bool api_keys_parse_failed = false;
    bool api_keys_empty_file = false;
    {
        const std::string api_keys_path =
            (std::filesystem::path(root) / "conf" / "api_keys.toml").string();
        if (std::filesystem::exists(api_keys_path)) {
            if (!g_api_keys.load(api_keys_path)) {
                api_keys_parse_failed = true;
                std::fprintf(stderr, "mortred-gateway: ERROR: failed to parse %s\n",
                             api_keys_path.c_str());
            } else if (g_api_keys.key_count() == 0) {
                api_keys_empty_file = true;
                std::fprintf(stderr,
                             "mortred-gateway: ERROR: %s parsed but contains no keys "
                             "(empty key file is not auth)\n",
                             api_keys_path.c_str());
            } else {
                api_keys_loaded = true;
                std::fprintf(stderr, "mortred-gateway: loaded %zu API keys from %s\n",
                             g_api_keys.key_count(), api_keys_path.c_str());
            }
        }
    }

    if (!jinq::common::listen_host_permitted(g_cfg.gateway.host)) {
        std::fprintf(stderr,
                     "mortred-gateway: refusing to listen on %s (MORTRED_EXPOSE=%s). "
                     "Bind 127.0.0.1 and terminate TLS at Nginx (deploy/nginx). "
                     "Containers: MORTRED_EXPOSE=docker. Metal wildcard: MORTRED_EXPOSE=unsafe "
                     "(plaintext; doctor --strict fails)\n",
                     g_cfg.gateway.host.c_str(),
                     jinq::common::mortred_expose_mode().c_str());
        return 1;
    }
    const bool has_static_token = !g_auth_token.empty() || !g_admin_token.empty();
    if (!has_static_token && !api_keys_loaded) {
        if (api_keys_empty_file) {
            std::fprintf(stderr,
                         "mortred-gateway: refusing to start: conf/api_keys.toml has no keys "
                         "(empty key file is not auth). Set MORTRED_GATEWAY_AUTH_TOKEN / "
                         "MORTRED_API_TOKEN, or add at least one [keys.*] hash. "
                         "Loopback is not an anonymous mode.\n");
        } else if (api_keys_parse_failed) {
            std::fprintf(stderr,
                         "mortred-gateway: refusing to start: conf/api_keys.toml exists but failed "
                         "to parse, and no MORTRED_GATEWAY_AUTH_TOKEN or MORTRED_API_TOKEN "
                         "fallback is configured\n");
        } else {
            std::fprintf(stderr,
                         "mortred-gateway: refusing to start without external auth "
                         "(set MORTRED_GATEWAY_AUTH_TOKEN, MORTRED_API_TOKEN, or provide a valid "
                         "conf/api_keys.toml). Loopback is not an anonymous mode.\n");
        }
        return 1;
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
    if (g_metrics_token.empty()) {
        std::fprintf(stderr,
                     "mortred-gateway: refusing to start without MORTRED_METRICS_TOKEN "
                     "(GET /metrics is never public, including loopback; set a scrape Bearer "
                     "distinct from the inference and management tokens). "
                     "Generate with: mortredctl init-trust\n");
        return 1;
    }
    if (g_metrics_token == g_auth_token || g_metrics_token == g_admin_token) {
        std::fprintf(stderr,
                     "mortred-gateway: refusing to start: MORTRED_METRICS_TOKEN matches an "
                     "inference or management token; Prometheus would then hold that privilege\n");
        return 1;
    }

    std::string catalog_err;
    const char* profile_env = std::getenv("MORTRED_PROFILE");
    const std::string runtime_profile =
        (profile_env != nullptr && std::string(profile_env) == "cpu") ? "cpu" : "gpu";
    if (!g_catalog.init(root, &catalog_err, runtime_profile)) {
        std::fprintf(stderr, "mortred-gateway: catalog init failed (profile=%s): %s\n",
                     runtime_profile.c_str(), catalog_err.c_str());
        return 1;
    }
    g_metrics.set_model("gateway");

    jinq::common::ProcessStop process_stop;
    process_stop.arm();

    WFServerParams params = SERVER_PARAMS_DEFAULT;
    params.max_connections = g_cfg.gateway.max_connections;
    params.request_size_limit =
        static_cast<size_t>(g_cfg.gateway.request_size_limit_mb) * 1024 * 1024;
    WFHttpServer server(&params, process);
    if (server.start(g_cfg.gateway.host.c_str(),
                     static_cast<unsigned short>(g_cfg.gateway.port)) != 0) {
        std::fprintf(stderr, "mortred-gateway: cannot listen on %s:%d\n",
                     g_cfg.gateway.host.c_str(), g_cfg.gateway.port);
        return 1;
    }
    const char* auth_mode = g_api_keys.key_count() > 0
                                ? "api-keys auth"
                                : "static-token auth";
    std::fprintf(stderr,
                 "mortred-gateway listening on http://%s:%d (routes: %zu) [%s, metrics scrape "
                 "token, expose=%s]\n",
                 g_cfg.gateway.host.c_str(), g_cfg.gateway.port, g_catalog.entries().size(),
                 auth_mode, jinq::common::mortred_expose_mode().c_str());

    process_stop.wait();
    server.stop();
    return 0;
}

}  // namespace control
}  // namespace mortred
