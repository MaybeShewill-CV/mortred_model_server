/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: main.cpp (mortred-supervisor)
* Date: 26-8-22
************************************************/

// Control-plane daemon: supervises mortred-gateway + all model servers,
// serves the /api/v1 management REST + web console (static file UI from
// share/mortred/ui, logs) and the embedded web UI. Inference goes through
// the gateway. All state is instance-local (SupervisorApp).

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
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
#include <vector>

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
#include "common/listen_policy.h"
#include "common/request_size_limit.h"
#include "control/catalog.h"
#include "control/control_config.h"
#include "control/http_reply.h"
#include "control/mini_toml.h"
#include "control/project_root.h"
#include "control/supervisor.h"

#include "control/supervisor/supervisor_app.h"

namespace {

using mortred::control::ProcessSupervisor;
using mortred::control::kGatewayId;

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

SupervisorApp::~SupervisorApp() {
    stop_listen();
}

void SupervisorApp::handle_catalog(WFHttpTask* task) {
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();
    rapidjson::Value servers(rapidjson::kArrayType);
    for (const auto& e : catalog_.entries()) {
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

void SupervisorApp::handle_status(WFHttpTask* task) {
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();

    const auto gateway_status = supervisor_->status(kGatewayId);
    rapidjson::Value gateway(rapidjson::kObjectType);
    add_status_object(a, &gateway, kGatewayId, gateway_status);
    rapidjson::Value gateway_addr(rapidjson::kObjectType);
    gateway_addr.AddMember("host",
                           rapidjson::Value(cfg_.gateway.host.c_str(),
                                             cfg_.gateway.host.size(), a),
                           a);
    gateway_addr.AddMember("port", cfg_.gateway.port, a);
    gateway.AddMember("address", gateway_addr, a);
    d.AddMember("gateway", gateway, a);

    rapidjson::Value servers(rapidjson::kArrayType);
    for (const auto& [id, s] : supervisor_->statuses()) {
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

void SupervisorApp::handle_server_detail(WFHttpTask* task, const std::string& id) {
    if (!supervisor_->has_server(id)) {
        reply_json(task, 404, json_error("unknown server id: " + id));
        return;
    }
    const auto s = supervisor_->status(id);
    rapidjson::Document d;
    d.SetObject();
    auto& a = d.GetAllocator();
    rapidjson::Value obj(rapidjson::kObjectType);
    add_status_object(a, &obj, id, s);
    const auto* entry = catalog_.find(id);
    if (entry != nullptr) {
        obj.AddMember("uri", rapidjson::Value(entry->uri.c_str(), entry->uri.size(), a), a);
        obj.AddMember("port", entry->port, a);
        obj.AddMember("category",
                      rapidjson::Value(entry->category.c_str(), entry->category.size(), a), a);
    }
    d.AddMember("server", obj, a);
    reply_json(task, 200, serialize(d));
}

void SupervisorApp::handle_server_action(WFHttpTask* task, const std::string& id,
                                         const std::string& action) {
    if (!supervisor_->has_server(id)) {
        reply_json(task, 404, json_error("unknown server id: " + id));
        return;
    }
    std::string err;
    bool ok = false;
    if (action == "start") {
        ok = supervisor_->start_server(id, &err);
    } else if (action == "stop") {
        ok = supervisor_->stop_server(id, &err);
    } else if (action == "restart") {
        ok = supervisor_->restart_server(id, &err);
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

void SupervisorApp::handle_logs(WFHttpTask* task, const std::string& id,
                                const std::string& uri) {
    auto* buffer = supervisor_->logs(id);
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

void SupervisorApp::handle_metrics(WFHttpTask* task) {
    std::ostringstream ss;
    ss << "# HELP mortred_supervisor_state Supervised process state (0=stopped,1=starting,"
          "2=running,3=backoff,4=failed)\n";
    ss << "# TYPE mortred_supervisor_state gauge\n";
    for (const auto& [id, s] : supervisor_->statuses()) {
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
    for (const auto& [id, s] : supervisor_->statuses()) {
        ss << "mortred_supervisor_ready{server=\"" << id << "\"} " << (s.ready ? 1 : 0) << "\n";
    }
    ss << "# HELP mortred_supervisor_restarts_total Total restarts per supervised process\n";
    ss << "# TYPE mortred_supervisor_restarts_total counter\n";
    for (const auto& [id, s] : supervisor_->statuses()) {
        ss << "mortred_supervisor_restarts_total{server=\"" << id << "\"} " << s.restart_count
           << "\n";
    }
    auto* resp = task->get_resp();
    resp->set_status_code("200");
    resp->add_header_pair("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
    const auto body = ss.str();
    resp->append_output_body(body.data(), body.size());
}

void SupervisorApp::serve_static(WFHttpTask* task, const std::string& path) {
    std::string rel = (path == "/" || path.empty()) ? "index.html" : path.substr(1);
    if (rel.find("..") != std::string::npos) {
        reply_json(task, 400, json_error("bad path"));
        return;
    }
    const std::string file = (std::filesystem::path(ui_dir_) / rel).string();
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

/*** check the model's loopback /metrics for in-flight async jobs (graceful drain) */
bool SupervisorApp::server_has_active_jobs(const std::string& server_id) {
    const auto* entry = catalog_.find(server_id);
    if (entry == nullptr) {
        return false;
    }
    const auto s = supervisor_->status(server_id);
    if (s.pid < 0) {
        return false;
    }
    const std::string url = "http://127.0.0.1:" + std::to_string(entry->port) + "/metrics";
    std::string body;
    int http_code = 0;
    WFFacilities::WaitGroup wg(1);
    auto* client = WFTaskFactory::create_http_task(
        url, 0, 0, [&wg, &body, &http_code](WFHttpTask* t) {
            if (t->get_state() == WFT_STATE_SUCCESS) {
                http_code = std::atoi(t->get_resp()->get_status_code());
                const void* data = nullptr;
                size_t size = 0;
                t->get_resp()->get_parsed_body(&data, &size);
                if (data != nullptr && size > 0) {
                    body.assign(static_cast<const char*>(data), size);
                }
            }
            wg.done();
        });
    client->get_req()->set_method("GET");
    std::string metrics_auth;
    if (supervisor_ != nullptr && !supervisor_->internal_token().empty()) {
        metrics_auth = "Bearer " + supervisor_->internal_token();
        client->get_req()->add_header_pair("Authorization", metrics_auth.c_str());
    }
    client->set_receive_timeout(2000);
    client->start();
    wg.wait();
    if (http_code != 200) {
        // cannot observe the ledger: wait rather than restart into in-flight jobs
        return true;
    }
    std::istringstream in(body);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind("mortred_async_queue_depth", 0) != 0) {
            continue;
        }
        const auto sp = line.rfind(' ');
        if (sp == std::string::npos) {
            continue;
        }
        if (std::atof(line.c_str() + sp + 1) > 0.0) {
            return true;
        }
    }
    return false;
}

/*** graceful drain: wait up to timeout for async jobs to complete before restart */
void SupervisorApp::handle_graceful_restart(WFHttpTask* task, const std::string& server_id) {
    constexpr int k_drain_timeout_ms = 120000;  // 2 min
    constexpr int k_poll_interval_ms = 2000;

    auto* series = series_of(task);
    auto* go = WFTaskFactory::create_go_task(
        "graceful_restart", [this, task, server_id, k_drain_timeout_ms, k_poll_interval_ms]() {
            const auto start = std::chrono::steady_clock::now();
            bool drained = false;
            while (std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count() < k_drain_timeout_ms) {
                if (!server_has_active_jobs(server_id)) {
                    drained = true;
                    break;
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(k_poll_interval_ms));
            }
            std::string err;
            const bool ok = supervisor_->restart_server(server_id, &err);
            auto* resp = task->get_resp();
            resp->set_status_code(ok ? "200" : "500");
            resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
            rapidjson::Document d;
            d.SetObject();
            auto& a = d.GetAllocator();
            d.AddMember("ok", ok, a);
            d.AddMember("drained", drained, a);
            if (!err.empty()) {
                d.AddMember("error", rapidjson::Value(err.c_str(), err.size(), a), a);
            }
            rapidjson::StringBuffer buf;
            rapidjson::Writer<rapidjson::StringBuffer> w(buf);
            d.Accept(w);
            resp->append_output_body(buf.GetString(), buf.GetSize());
        });
    series->push_back(go);
}

void SupervisorApp::process(WFHttpTask* task) {
    const std::string path = uri_path(task->get_req()->get_request_uri());
    // workflow leaves method null when the request line is malformed; fold to
    // "" like the gateway/model-server guards instead of UB in std::string
    const char* raw_method = task->get_req()->get_method();
    const std::string method = raw_method == nullptr ? "" : raw_method;
    const std::string full_uri =
        task->get_req()->get_request_uri() == nullptr ? "" : task->get_req()->get_request_uri();

    const bool is_api = path.rfind("/api/v1/", 0) == 0;
    // health stays public (docker healthcheck / k8s probes); the rest of the
    // management API requires the supervisor bearer token
    if (is_api && path != "/api/v1/health" &&
        !jinq::common::is_bearer_authorized(
            header_value(task->get_req(), "Authorization"), auth_token_)) {
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
            if (action == "graceful_restart") {
                handle_graceful_restart(task, id);
                return;
            }
            handle_server_action(task, id, action);
        } else {
            reply_json(task, 405, json_error("method not allowed"));
        }
    } else {
        reply_json(task, 404, json_error("not found"));
    }
}

bool SupervisorApp::init(const SupervisorInitOptions& options) {
    root_ = options.project_root;
    std::string config_path = options.config_path;
    if (config_path.empty()) {
        config_path = (std::filesystem::path(root_) / "conf" / "mortred.toml").string();
    }
    std::string cfg_err;
    if (!ControlConfig::load(config_path, &cfg_, &cfg_err)) {
        std::fprintf(stderr, "mortred-supervisor: invalid control config: %s\n", cfg_err.c_str());
        return false;
    }

    if (!options.api_host.empty()) {
        cfg_.supervisor.api_host = options.api_host;
    }
    if (options.api_port > 0) {
        cfg_.supervisor.api_port = options.api_port;
    }
    auth_token_ = options.api_token;
    if (options.autostart_default >= 0) {
        cfg_.supervisor.autostart_default = options.autostart_default != 0;
    }
    if (!options.bin_dir.empty()) {
        cfg_.supervisor.bin_dir = options.bin_dir;
    }
    if (!options.lib_dir.empty()) {
        cfg_.supervisor.lib_dir = options.lib_dir;
    }
    if (!options.libs_dir.empty()) {
        cfg_.supervisor.libs_dir = options.libs_dir;
    }
    if (!options.ui_dir.empty()) {
        ui_dir_ = options.ui_dir;
    } else {
        const std::filesystem::path install_ui = std::filesystem::path(root_) / "share" /
                                                  "mortred" / "ui";
        const std::filesystem::path source_ui = std::filesystem::path(root_) / "src" /
                                                 "control" / "supervisor" / "ui";
        std::error_code ec;
        ui_dir_ = std::filesystem::exists(install_ui / "index.html", ec)
                      ? install_ui.string()
                      : source_ui.string();
    }

    // fail-closed: management API always requires a token, including loopback
    if (auth_token_.empty()) {
        std::fprintf(stderr,
                     "mortred-supervisor: refusing to start without MORTRED_API_TOKEN "
                     "(loopback is not an anonymous management plane). "
                     "Generate with: mortredctl init-trust\n");
        return false;
    }
    if (!jinq::common::listen_host_permitted(cfg_.supervisor.api_host)) {
        std::fprintf(stderr,
                     "mortred-supervisor: refusing to listen on %s (MORTRED_EXPOSE=%s). "
                     "Bind 127.0.0.1; containers: MORTRED_EXPOSE=docker\n",
                     cfg_.supervisor.api_host.c_str(),
                     jinq::common::mortred_expose_mode().c_str());
        return false;
    }

    std::string runtime_profile = options.profile;
    if (runtime_profile.empty()) {
        const char* profile_env = std::getenv("MORTRED_PROFILE");
        runtime_profile =
            (profile_env != nullptr && std::string(profile_env) == "cpu") ? "cpu" : "gpu";
    }
    std::string catalog_err;
    if (!catalog_.init(root_, &catalog_err, runtime_profile)) {
        std::fprintf(stderr, "mortred-supervisor: catalog init failed (profile=%s): %s\n",
                     runtime_profile.c_str(), catalog_err.c_str());
        return false;
    }

    std::string pack_path = options.pack_path;
    if (!pack_path.empty()) {
        std::filesystem::path pack(pack_path);
        if (!pack.is_absolute()) {
            pack = std::filesystem::path(root_) / pack;
        }
        std::vector<std::string> catalog_ids;
        catalog_ids.reserve(catalog_.entries().size());
        for (const auto& e : catalog_.entries()) {
            catalog_ids.push_back(e.id);
        }
        std::string pack_err;
        if (!ControlConfig::apply_pack(pack.string(), catalog_ids, root_, &cfg_, &pack_err)) {
            std::fprintf(stderr, "mortred-supervisor: invalid pack: %s\n", pack_err.c_str());
            return false;
        }
        size_t pack_n = 0;
        for (const auto& item : cfg_.servers) {
            if (item.second.has_autostart && item.second.autostart) {
                ++pack_n;
            }
        }
        std::fprintf(stderr, "mortred-supervisor: pack %s (autostart %zu model(s))\n",
                     pack.string().c_str(), pack_n);
    }

    supervisor_ = std::make_unique<ProcessSupervisor>(root_, cfg_, config_path);
    supervisor_->set_catalog(catalog_);
    std::string thread_err;
    if (!supervisor_->start_threads(&thread_err)) {
        std::fprintf(stderr, "mortred-supervisor: %s\n", thread_err.c_str());
        return false;
    }
    return true;
}

bool SupervisorApp::listen() {
    WFServerParams params = SERVER_PARAMS_DEFAULT;
    params.request_size_limit =
        jinq::common::k_default_request_size_limit_mb * 1024 * 1024;
    server_ = std::make_unique<WFHttpServer>(
        &params, [this](WFHttpTask* task) { process(task); });
    if (server_->start(cfg_.supervisor.api_host.c_str(),
                       static_cast<unsigned short>(cfg_.supervisor.api_port)) != 0) {
        std::fprintf(stderr, "mortred-supervisor: cannot listen on %s:%d\n",
                     cfg_.supervisor.api_host.c_str(), cfg_.supervisor.api_port);
        server_.reset();
        return false;
    }
    return true;
}

void SupervisorApp::stop_listen() {
    if (server_ != nullptr) {
        // WFServerBase::stop() is ALREADY shutdown()+wait_finish() (blocking).
        // The historical "stop(); wait_finish();" double call hung forever on
        // the second wait - in the daemon this was masked by systemd's
        // TimeoutStopSec SIGKILL, so mortred-supervisor never actually exited
        // gracefully. stop() alone is the correct, complete teardown.
        server_->stop();
        server_.reset();
    }
    if (supervisor_ != nullptr) {
        supervisor_->request_shutdown();
        supervisor_->wait_shutdown();
    }
}

int SupervisorApp::run() {
    // supervision signals must be blocked before any thread exists
    ProcessSupervisor::block_supervision_signals();

    SupervisorInitOptions options;
    options.project_root = resolve_project_root();
    if (const char* env = std::getenv("MORTRED_CONTROL_CONFIG");
        env != nullptr && *env != '\0') {
        options.config_path = env;
    }
    if (const char* env = std::getenv("MORTRED_API_HOST"); env != nullptr && *env != '\0') {
        options.api_host = env;
    }
    options.api_port = run_env_int("MORTRED_API_PORT", 0);
    if (const char* env = std::getenv("MORTRED_API_TOKEN"); env != nullptr && *env != '\0') {
        options.api_token = env;
    }
    if (const char* env = std::getenv("MORTRED_AUTOSTART"); env != nullptr && *env != '\0') {
        const std::string v = mini_toml::trim(env);
        options.autostart_default = (v == "true" || v == "1") ? 1 : 0;
    }
    if (const char* env = std::getenv("APP_BIN_DIR"); env != nullptr && *env != '\0') {
        options.bin_dir = env;
    }
    if (const char* env = std::getenv("APP_LIB_DIR"); env != nullptr && *env != '\0') {
        options.lib_dir = env;
    }
    if (const char* env = std::getenv("APP_LIBS_DIR"); env != nullptr && *env != '\0') {
        options.libs_dir = env;
    }
    if (const char* env = std::getenv("MORTRED_UI_DIR"); env != nullptr && *env != '\0') {
        options.ui_dir = env;
    }
    if (const char* env = std::getenv("MORTRED_PACK"); env != nullptr && *env != '\0') {
        options.pack_path = env;
    }
    if (!init(options)) {
        return 1;
    }

    // Gateway reads these at startup. Inject before autostart so the child
    // inherits them: management token is a valid data-plane credential for UI
    // / mortredctl infer, and CORS origins let the :8787 UI call :8080.
    // (The process environment is deliberately global: children inherit it.)
    if (!auth_token_.empty()) {
        ::setenv("MORTRED_API_TOKEN", auth_token_.c_str(), 1);
    }
    {
        const std::string port = std::to_string(cfg_.supervisor.api_port);
        std::string origins = "http://127.0.0.1:" + port + ",http://localhost:" + port;
        const std::string& host = cfg_.supervisor.api_host;
        if (host != "0.0.0.0" && host != "::" && host != "[::]" && host != "127.0.0.1" &&
            host != "localhost") {
            origins += ",http://" + host + ":" + port;
        }
        ::setenv("MORTRED_GATEWAY_CORS_ORIGINS", origins.c_str(), 1);
    }

    if (!listen()) {
        return 1;
    }
    std::fprintf(stderr,
                 "mortred-supervisor listening on http://%s:%d (managed servers: %zu, auth enabled, "
                 "expose=%s)\n",
                 cfg_.supervisor.api_host.c_str(), cfg_.supervisor.api_port,
                 catalog_.entries().size(), jinq::common::mortred_expose_mode().c_str());

    supervisor_->autostart_all();

    while (!supervisor_->shutdown_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    stop_listen();
    return 0;
}

int run_supervisor() {
    return SupervisorApp().run();
}

}  // namespace control
}  // namespace mortred
