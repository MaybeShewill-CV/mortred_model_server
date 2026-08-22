/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: main.cpp (mortred-supervisor)
* Date: 26-8-22
************************************************/

// Control-plane daemon: supervises mortred-gateway + all model servers,
// exposes the versioned /api/v1 management REST API and serves the embedded
// web UI. Single supervision tree: systemd/container starts this process, it
// starts everything else and stops it in reverse order on SIGINT/SIGTERM.

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <workflow/HttpMessage.h>
#include <workflow/HttpUtil.h>
#include <workflow/WFFacilities.h>
#include <workflow/WFHttpServer.h>
#include <workflow/WFTaskFactory.h>
#include <workflow/Workflow.h>

#include "common/auth_token.h"
#include "common/request_size_limit.h"
#include "control/catalog.h"
#include "control/control_config.h"
#include "control/mini_toml.h"
#include "control/supervisor.h"

#include "control/supervisor/supervisor_app.h"

namespace {

using mortred::control::Catalog;
using mortred::control::ControlConfig;
using mortred::control::ProcessSupervisor;
using mortred::control::kGatewayId;

Catalog g_catalog;
std::unique_ptr<ProcessSupervisor> g_supervisor;
ControlConfig g_cfg;
std::string g_root;
std::string g_ui_dir;
std::string g_auth_token;

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

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return "";
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string content_type_for(const std::string& path) {
    auto ends = [&path](const char* suffix) {
        const size_t n = std::strlen(suffix);
        return path.size() > n && path.compare(path.size() - n, n, suffix) == 0;
    };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".js")) return "application/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".png")) return "image/png";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".ico")) return "image/x-icon";
    return "application/octet-stream";
}

std::string uri_path(const char* uri) {
    const std::string s(uri == nullptr ? "" : uri);
    const auto q = s.find('?');
    return q == std::string::npos ? s : s.substr(0, q);
}

std::string query_value(const std::string& uri, const std::string& key) {
    const auto q = uri.find('?');
    if (q == std::string::npos) {
        return "";
    }
    const std::string query = uri.substr(q + 1);
    size_t pos = 0;
    while (pos < query.size()) {
        const auto amp = query.find('&', pos);
        const std::string pair =
            query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        const auto eq = pair.find('=');
        if (eq != std::string::npos && pair.substr(0, eq) == key) {
            return pair.substr(eq + 1);
        }
        if (amp == std::string::npos) {
            break;
        }
        pos = amp + 1;
    }
    return "";
}

size_t parse_size(const std::string& s, size_t fallback) {
    if (s.empty()) {
        return fallback;
    }
    try {
        return static_cast<size_t>(std::stoull(s));
    } catch (...) {
        return fallback;
    }
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

std::string json_error(const std::string& msg) {
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();
    d.AddMember("ok", false, a);
    d.AddMember("error", rapidjson::Value(msg.c_str(), msg.size(), a), a);
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> w(buf);
    d.Accept(w);
    return buf.GetString();
}

std::string serialize(const rapidjson::Document& d) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> w(buf);
    d.Accept(w);
    return buf.GetString();
}

void add_status_object(rapidjson::Document::AllocatorType& a, rapidjson::Value* obj,
                       const std::string& id, const ProcessSupervisor::Status& s) {
    obj->SetObject();
    obj->AddMember("id", rapidjson::Value(id.c_str(), id.size(), a), a);
    obj->AddMember("state", rapidjson::Value(s.state.c_str(), s.state.size(), a), a);
    obj->AddMember("pid", s.pid, a);
    obj->AddMember("ready", s.ready, a);
    obj->AddMember("restart_count", s.restart_count, a);
    if (s.has_last_exit) {
        obj->AddMember("last_exit_status", s.last_exit_status, a);
    }
    obj->AddMember("started_at_ms", s.started_at_unix_ms, a);
    if (!s.error.empty()) {
        obj->AddMember("error", rapidjson::Value(s.error.c_str(), s.error.size(), a), a);
    }
}

void handle_catalog(WFHttpTask* task) {
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();
    rapidjson::Value servers(rapidjson::kArrayType);
    for (const auto& e : g_catalog.entries()) {
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("id", rapidjson::Value(e.id.c_str(), e.id.size(), a), a);
        obj.AddMember("name", rapidjson::Value(e.name.c_str(), e.name.size(), a), a);
        obj.AddMember("category", rapidjson::Value(e.category.c_str(), e.category.size(), a), a);
        obj.AddMember("type", rapidjson::Value(e.type.c_str(), e.type.size(), a), a);
        obj.AddMember("uri", rapidjson::Value(e.uri.c_str(), e.uri.size(), a), a);
        obj.AddMember("port", e.port, a);
        servers.PushBack(obj, a);
    }
    d.AddMember("servers", servers, a);
    reply_json(task, 200, serialize(d));
}

void handle_status(WFHttpTask* task) {
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();

    const auto gateway_status = g_supervisor->status(kGatewayId);
    rapidjson::Value gateway(rapidjson::kObjectType);
    add_status_object(a, &gateway, kGatewayId, gateway_status);
    rapidjson::Value gateway_addr(rapidjson::kObjectType);
    gateway_addr.AddMember("host",
                           rapidjson::Value(g_cfg.gateway.host.c_str(),
                                             g_cfg.gateway.host.size(), a),
                           a);
    gateway_addr.AddMember("port", g_cfg.gateway.port, a);
    gateway.AddMember("address", gateway_addr, a);
    d.AddMember("gateway", gateway, a);

    rapidjson::Value servers(rapidjson::kArrayType);
    for (const auto& [id, s] : g_supervisor->statuses()) {
        if (id == kGatewayId) {
            continue;
        }
        rapidjson::Value obj(rapidjson::kObjectType);
        add_status_object(a, &obj, id, s);
        servers.PushBack(obj, a);
    }
    d.AddMember("servers", servers, a);
    reply_json(task, 200, serialize(d));
}

void handle_server_detail(WFHttpTask* task, const std::string& id) {
    if (!g_supervisor->has_server(id)) {
        reply_json(task, 404, json_error("unknown server id: " + id));
        return;
    }
    const auto s = g_supervisor->status(id);
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();
    rapidjson::Value obj(rapidjson::kObjectType);
    add_status_object(a, &obj, id, s);
    const auto* entry = g_catalog.find(id);
    if (entry != nullptr) {
        obj.AddMember("uri", rapidjson::Value(entry->uri.c_str(), entry->uri.size(), a), a);
        obj.AddMember("port", entry->port, a);
        obj.AddMember("category",
                      rapidjson::Value(entry->category.c_str(), entry->category.size(), a), a);
    }
    d.AddMember("server", obj, a);
    reply_json(task, 200, serialize(d));
}

void handle_server_action(WFHttpTask* task, const std::string& id, const std::string& action) {
    if (!g_supervisor->has_server(id)) {
        reply_json(task, 404, json_error("unknown server id: " + id));
        return;
    }
    std::string err;
    bool ok = false;
    if (action == "start") {
        ok = g_supervisor->start_server(id, &err);
    } else if (action == "stop") {
        ok = g_supervisor->stop_server(id, &err);
    } else if (action == "restart") {
        ok = g_supervisor->restart_server(id, &err);
    } else {
        reply_json(task, 400, json_error("unknown action: " + action));
        return;
    }
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();
    d.AddMember("ok", ok, a);
    if (!err.empty()) {
        d.AddMember("error", rapidjson::Value(err.c_str(), err.size(), a), a);
    }
    reply_json(task, 200, serialize(d));
}

void handle_logs(WFHttpTask* task, const std::string& id, const std::string& uri) {
    auto* buffer = g_supervisor->logs(id);
    if (buffer == nullptr) {
        reply_json(task, 404, json_error("unknown server id: " + id));
        return;
    }
    size_t offset = parse_size(query_value(uri, "offset"), 0);
    size_t limit = parse_size(query_value(uri, "limit"), 200);
    if (limit > 1000) {
        limit = 1000;
    }
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();
    d.AddMember("offset", static_cast<uint64_t>(offset), a);
    d.AddMember("total", static_cast<uint64_t>(buffer->size()), a);
    rapidjson::Value lines(rapidjson::kArrayType);
    for (const auto& line : buffer->slice(offset, limit)) {
        lines.PushBack(rapidjson::Value(line.c_str(), line.size(), a), a);
    }
    d.AddMember("lines", lines, a);
    reply_json(task, 200, serialize(d));
}

void handle_metrics(WFHttpTask* task) {
    std::ostringstream ss;
    ss << "# HELP mortred_supervisor_state Supervised process state (0=stopped,1=starting,"
          "2=running,3=backoff,4=failed)\n";
    ss << "# TYPE mortred_supervisor_state gauge\n";
    for (const auto& [id, s] : g_supervisor->statuses()) {
        int code = 0;
        if (s.state == "starting") {
            code = 1;
        } else if (s.state == "running") {
            code = 2;
        } else if (s.state == "backoff") {
            code = 3;
        } else if (s.state == "failed") {
            code = 4;
        }
        ss << "mortred_supervisor_state{server=\"" << id << "\"} " << code << "\n";
    }
    ss << "# HELP mortred_supervisor_ready Readiness of supervised processes\n";
    ss << "# TYPE mortred_supervisor_ready gauge\n";
    for (const auto& [id, s] : g_supervisor->statuses()) {
        ss << "mortred_supervisor_ready{server=\"" << id << "\"} " << (s.ready ? 1 : 0) << "\n";
    }
    ss << "# HELP mortred_supervisor_restarts_total Total restarts per supervised process\n";
    ss << "# TYPE mortred_supervisor_restarts_total counter\n";
    for (const auto& [id, s] : g_supervisor->statuses()) {
        ss << "mortred_supervisor_restarts_total{server=\"" << id << "\"} " << s.restart_count
           << "\n";
    }
    auto* resp = task->get_resp();
    resp->set_status_code("200");
    resp->add_header_pair("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
    const auto body = ss.str();
    resp->append_output_body(body.data(), body.size());
}

void handle_infer(WFHttpTask* task) {
    // management-plane test proxy: {server_id, img_data} -> model server.
    // Production traffic goes through mortred-gateway, not this endpoint.
    const std::string body = protocol::HttpUtil::decode_chunked_body(task->get_req());
    rapidjson::Document doc;
    doc.Parse(body.c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("server_id") ||
        !doc["server_id"].IsString()) {
        reply_json(task, 400, json_error("body must contain string field server_id"));
        return;
    }
    const std::string id = doc["server_id"].GetString();
    const auto* entry = g_catalog.find(id);
    const bool is_gateway = id == kGatewayId;
    if (entry == nullptr && !is_gateway) {
        reply_json(task, 404, json_error("unknown server id: " + id));
        return;
    }
    const auto s = g_supervisor->status(id);
    if (s.pid < 0) {
        reply_json(task, 409, json_error("server not running: " + id));
        return;
    }
    if (!s.ready) {
        reply_json(task, 409,
                   json_error("server not ready yet (loading or unhealthy): " + id));
        return;
    }
    const std::string url = is_gateway
                                ? "http://127.0.0.1:" + std::to_string(g_cfg.gateway.port) +
                                      "/healthz"
                                : "http://127.0.0.1:" + std::to_string(entry->port) + entry->uri;
    const std::string internal_token = g_supervisor->internal_token();
    auto* client = WFTaskFactory::create_http_task(
        url, 0, 0, [task](WFHttpTask* t) {
            auto* resp = task->get_resp();
            if (t->get_state() != WFT_STATE_SUCCESS) {
                const int code = t->get_error() == ECONNREFUSED ? 503 : 502;
                resp->set_status_code(std::to_string(code).c_str());
                resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
                const std::string err = json_error("forward to model server failed");
                resp->append_output_body(err.data(), err.size());
                return;
            }
            resp->set_status_code(t->get_resp()->get_status_code());
            resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
            const void* data = nullptr;
            size_t size = 0;
            t->get_resp()->get_parsed_body(&data, &size);
            resp->append_output_body(data, size);
        });
    client->get_req()->set_method("POST");
    client->get_req()->add_header_pair("Content-Type", "application/json; charset=utf-8");
    if (!internal_token.empty()) {
        const std::string auth = "Bearer " + internal_token;
        client->get_req()->add_header_pair("Authorization", auth.c_str());
    }
    client->get_req()->append_output_body(body.data(), body.size());
    client->set_send_timeout(-1);
    client->set_receive_timeout(g_cfg.gateway.upstream_recv_timeout_ms);
    series_of(task)->push_back(client);
}

void serve_static(WFHttpTask* task, const std::string& path) {
    std::string rel = (path == "/" || path.empty()) ? "index.html" : path.substr(1);
    if (rel.find("..") != std::string::npos) {
        reply_json(task, 400, json_error("bad path"));
        return;
    }
    const std::string file = (std::filesystem::path(g_ui_dir) / rel).string();
    const std::string content = read_file(file);
    if (content.empty()) {
        reply_json(task, 404, json_error("not found"));
        return;
    }
    auto* resp = task->get_resp();
    resp->set_status_code("200");
    resp->add_header_pair("Content-Type", content_type_for(rel).c_str());
    resp->append_output_body(content.data(), content.size());
}

void process(WFHttpTask* task) {
    const std::string path = uri_path(task->get_req()->get_request_uri());
    const std::string method = task->get_req()->get_method();
    const std::string full_uri =
        task->get_req()->get_request_uri() == nullptr ? "" : task->get_req()->get_request_uri();

    const bool is_api = path.rfind("/api/v1/", 0) == 0;
    // health stays public (docker healthcheck / k8s probes); the rest of the
    // management API requires the supervisor bearer token
    if (is_api && path != "/api/v1/health" &&
        !jinq::common::is_bearer_authorized(
            header_value(task->get_req(), "Authorization"), g_auth_token)) {
        task->get_resp()->add_header_pair("WWW-Authenticate", "Bearer realm=\"Mortred\"");
        reply_json(task, 401, json_error("unauthorized"));
        return;
    }

    if (path == "/api/v1/health") {
        reply_json(task, 200, "{\"ok\":true}");
        return;
    }
    if (!is_api) {
        if (method == "GET" &&
            (path == "/" || path == "/index.html" || path == "/app.js" || path == "/style.css" ||
             path == "/favicon.ico")) {
            serve_static(task, path);
            return;
        }
        reply_json(task, 404, json_error("not found"));
        return;
    }

    if (path == "/api/v1/catalog" && method == "GET") {
        handle_catalog(task);
    } else if (path == "/api/v1/status" && method == "GET") {
        handle_status(task);
    } else if (path == "/api/v1/metrics" && method == "GET") {
        handle_metrics(task);
    } else if (path == "/api/v1/infer" && method == "POST") {
        handle_infer(task);
    } else if (path.rfind("/api/v1/servers/", 0) == 0) {
        const std::string rest = path.substr(std::string("/api/v1/servers/").size());
        const auto slash = rest.rfind('/');
        if (slash == std::string::npos) {
            if (method != "GET") {
                reply_json(task, 405, json_error("method not allowed"));
                return;
            }
            handle_server_detail(task, rest);
            return;
        }
        const std::string id = rest.substr(0, slash);
        const std::string action = rest.substr(slash + 1);
        if (action == "logs") {
            if (method != "GET") {
                reply_json(task, 405, json_error("method not allowed"));
                return;
            }
            handle_logs(task, id, full_uri);
        } else if (method == "POST") {
            handle_server_action(task, id, action);
        } else {
            reply_json(task, 405, json_error("method not allowed"));
        }
    } else {
        reply_json(task, 404, json_error("not found"));
    }
}

int run_env_int(const char* name, int fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return fallback;
    }
    try {
        return std::stoi(env);
    } catch (...) {
        return fallback;
    }
}

}  // namespace

namespace mortred {
namespace control {

int run_supervisor() {
    // supervision signals must be blocked before any thread exists
    ProcessSupervisor::block_supervision_signals();

    g_root = resolve_project_root();
    std::string config_path;
    if (const char* env = std::getenv("MORTRED_CONTROL_CONFIG");
        env != nullptr && *env != '\0') {
        config_path = env;
    } else {
        config_path = (std::filesystem::path(g_root) / "conf" / "mortred.toml").string();
    }
    std::string cfg_err;
    if (!ControlConfig::load(config_path, &g_cfg, &cfg_err)) {
        std::fprintf(stderr, "mortred-supervisor: invalid control config: %s\n", cfg_err.c_str());
        return 1;
    }

    if (const char* env = std::getenv("MORTRED_API_HOST"); env != nullptr && *env != '\0') {
        g_cfg.supervisor.api_host = env;
    }
    g_cfg.supervisor.api_port = run_env_int("MORTRED_API_PORT", g_cfg.supervisor.api_port);
    if (const char* env = std::getenv("MORTRED_API_TOKEN"); env != nullptr && *env != '\0') {
        g_auth_token = env;
    }
    if (const char* env = std::getenv("MORTRED_AUTOSTART"); env != nullptr && *env != '\0') {
        const std::string v = mortred::control::mini_toml::trim(env);
        g_cfg.supervisor.autostart_default = (v == "true" || v == "1");
    }
    if (const char* env = std::getenv("APP_BIN_DIR"); env != nullptr && *env != '\0') {
        g_cfg.supervisor.bin_dir = env;
    }
    if (const char* env = std::getenv("APP_LIB_DIR"); env != nullptr && *env != '\0') {
        g_cfg.supervisor.lib_dir = env;
    }
    if (const char* env = std::getenv("APP_LIBS_DIR"); env != nullptr && *env != '\0') {
        g_cfg.supervisor.libs_dir = env;
    }
    if (const char* env = std::getenv("MORTRED_UI_DIR"); env != nullptr && *env != '\0') {
        g_ui_dir = env;
    } else {
        const std::filesystem::path install_ui = std::filesystem::path(g_root) / "share" / "mortred" / "ui";
        const std::filesystem::path source_ui =
            std::filesystem::path(g_root) / "src" / "control" / "supervisor" / "ui";
        std::error_code ec;
        g_ui_dir = std::filesystem::exists(install_ui / "index.html", ec)
                       ? install_ui.string()
                       : source_ui.string();
    }

    // fail-closed: non-loopback management API requires a token
    if (!jinq::common::is_loopback_host(g_cfg.supervisor.api_host) && g_auth_token.empty()) {
        std::fprintf(stderr,
                     "mortred-supervisor: refusing to listen on non-loopback host %s without "
                     "MORTRED_API_TOKEN\n",
                     g_cfg.supervisor.api_host.c_str());
        return 1;
    }

    std::string catalog_err;
    if (!g_catalog.init(g_root, &catalog_err)) {
        std::fprintf(stderr, "mortred-supervisor: catalog init failed: %s\n",
                     catalog_err.c_str());
        return 1;
    }

    g_supervisor = std::make_unique<ProcessSupervisor>(g_root, g_cfg, config_path);
    g_supervisor->set_catalog(g_catalog);
    std::string thread_err;
    if (!g_supervisor->start_threads(&thread_err)) {
        std::fprintf(stderr, "mortred-supervisor: %s\n", thread_err.c_str());
        return 1;
    }

    WFServerParams params = SERVER_PARAMS_DEFAULT;
    params.request_size_limit =
        jinq::common::k_default_request_size_limit_mb * 1024 * 1024;
    WFHttpServer server(&params, process);
    if (server.start(g_cfg.supervisor.api_host.c_str(),
                     static_cast<unsigned short>(g_cfg.supervisor.api_port)) != 0) {
        std::fprintf(stderr, "mortred-supervisor: cannot listen on %s:%d\n",
                     g_cfg.supervisor.api_host.c_str(), g_cfg.supervisor.api_port);
        return 1;
    }
    std::fprintf(stderr,
                 "mortred-supervisor listening on http://%s:%d (managed servers: %zu)%s\n",
                 g_cfg.supervisor.api_host.c_str(), g_cfg.supervisor.api_port,
                 g_catalog.entries().size(),
                 g_auth_token.empty() ? " (auth disabled)" : " (auth enabled)");

    g_supervisor->autostart_all();

    while (!g_supervisor->shutdown_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    server.stop();
    server.wait_finish();
    return 0;
}

}  // namespace control
}  // namespace mortred
