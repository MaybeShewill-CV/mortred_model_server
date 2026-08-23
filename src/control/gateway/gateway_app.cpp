/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: main.cpp (mortred-gateway)
* Date: 26-8-22
************************************************/

// Data-plane reverse proxy: single external entry that routes each model
// server_uri to its loopback model server. Stateless by design; the
// supervisor owns its lifecycle (restart on exit, config reload = restart).

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

#include <workflow/HttpMessage.h>
#include <workflow/HttpUtil.h>
#include <workflow/WFFacilities.h>
#include <workflow/WFHttpServer.h>
#include <workflow/WFTaskFactory.h>
#include <workflow/Workflow.h>

#include "common/auth_token.h"
#include "control/api_key_manager.h"
#include "control/catalog.h"
#include "control/control_config.h"
#include "server/prometheus_metrics.h"

#include "control/gateway/gateway_app.h"

namespace {

using mortred::control::Catalog;
using mortred::control::ControlConfig;

Catalog g_catalog;
ControlConfig g_cfg;
std::string g_auth_token;       // external bearer token ("" = loopback mode)
mortred::control::ApiKeyManager g_api_keys;  // multi-key auth (P0-4)
std::string g_internal_token;   // shared with model servers via supervisor env
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

void reply_json(WFHttpTask* task, int http_status, const std::string& body) {
    auto* resp = task->get_resp();
    resp->set_status_code(std::to_string(http_status).c_str());
    resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
    resp->add_header_pair("Cache-Control", "no-store");
    resp->append_output_body(body.data(), body.size());
}

void reply_error(WFHttpTask* task, int http_status, const std::string& msg) {
    // fixed internal messages only: no client data is reflected into JSON
    reply_json(task, http_status, "{\"error\":\"" + msg + "\"}");
}

std::string uri_path(const char* uri) {
    const std::string s(uri == nullptr ? "" : uri);
    const auto q = s.find('?');
    return q == std::string::npos ? s : s.substr(0, q);
}

void forward_to_model(WFHttpTask* task, const mortred::control::ServerEntry& entry) {
    const std::string method = task->get_req()->get_method();
    if (method != "POST") {
        g_metrics.inc_http_requests(method, "405");
        task->get_resp()->add_header_pair("Allow", "POST");
        reply_error(task, 405, "method not allowed");
        return;
    }

    std::string body = protocol::HttpUtil::decode_chunked_body(task->get_req());
    const std::string url = "http://127.0.0.1:" + std::to_string(entry.port) + entry.uri;
    const std::string route = entry.id;
    const int send_timeout = g_cfg.gateway.upstream_send_timeout_ms;
    const int recv_timeout = g_cfg.gateway.upstream_recv_timeout_ms;
    const std::string internal_token = g_internal_token;
    const auto t0 = std::chrono::steady_clock::now();

    auto* client = WFTaskFactory::create_http_task(
        url, 0, 0, [task, route, method, t0](WFHttpTask* t) {
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
                            code == 503 ? "model server not running or still loading"
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
            resp->add_header_pair("X-Mortred-Model", route.c_str());
            // forward upstream overload hints (429 + Retry-After) verbatim
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
                }
            }
            const void* data = nullptr;
            size_t size = 0;
            t->get_resp()->get_parsed_body(&data, &size);
            resp->append_output_body(data, size);
        });
    client->get_req()->set_method("POST");
    client->get_req()->append_output_body(body.data(), body.size());
    client->get_req()->add_header_pair("Content-Type", "application/json; charset=utf-8");
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
    const std::string method = task->get_req()->get_method();

    if (path == "/healthz") {
        task->get_resp()->set_status_code("200");
        task->get_resp()->add_header_pair("Content-Type", "text/plain; charset=utf-8");
        task->get_resp()->append_output_body("ok\n", 3);
        return;
    }
    if (path == "/metrics") {
        auto* resp = task->get_resp();
        resp->set_status_code("200");
        resp->add_header_pair("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        const auto body = g_metrics.render();
        resp->append_output_body(body.data(), body.size());
        return;
    }

    const auto* entry = g_catalog.find_by_uri(path);
    if (entry == nullptr) {
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
        if (key != nullptr && mortred::control::ApiKeyManager::has_scope(key, "inference")) {
            authorized = true;
            key_name = key->name;
        }
    }

    // fallback to single static token (legacy compatibility)
    if (!authorized &&
        jinq::common::is_bearer_authorized(auth_header, g_auth_token)) {
        authorized = true;
        key_name = "legacy";
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
    forward_to_model(task, *entry);
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
    if (const char* env = std::getenv("MORTRED_INTERNAL_TOKEN"); env != nullptr && *env != '\0') {
        g_internal_token = env;
    }

    // fail-closed: a non-loopback listener without an external token refuses to start
    if (!jinq::common::is_loopback_host(g_cfg.gateway.host) && g_auth_token.empty()) {
        std::fprintf(stderr,
                     "mortred-gateway: refusing to listen on non-loopback host %s without "
                     "MORTRED_GATEWAY_AUTH_TOKEN\n",
                     g_cfg.gateway.host.c_str());
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

    // multi-key auth: load from conf/api_keys.toml if present (P0-4)
    // falls back to the single static token if the file doesn't exist
    {
        const std::string api_keys_path =
            (std::filesystem::path(root) / "conf" / "api_keys.toml").string();
        if (std::filesystem::exists(api_keys_path)) {
            if (g_api_keys.load(api_keys_path)) {
                std::fprintf(stderr, "mortred-gateway: loaded %zu API keys from %s\n",
                             g_api_keys.key_count(), api_keys_path.c_str());
            } else {
                std::fprintf(stderr,
                             "mortred-gateway: WARNING: failed to parse %s "
                             "(falling back to static token)\n",
                             api_keys_path.c_str());
            }
        }
    }

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
    std::fprintf(stderr, "mortred-gateway listening on http://%s:%d (routes: %zu)%s\n",
                 g_cfg.gateway.host.c_str(), g_cfg.gateway.port, g_catalog.entries().size(),
                 g_auth_token.empty() ? " (auth disabled)" : " (auth enabled)");

    WFFacilities::WaitGroup wait_group(1);
    wait_group.wait();
    server.stop();
    return 0;
}

}  // namespace control
}  // namespace mortred
