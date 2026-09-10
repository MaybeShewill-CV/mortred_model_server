/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: do_work_lifetime_unittest.cc
* Date: 26-9-7
************************************************/

// Lifetime of do_work vs BaseAiServerImpl destruction, including the
// production sync deadline graph (serve_process -> WFCounterTask on the
// HTTP series, detached create_go_task per item, request-level timer).
// Direct call_do_work covers destructor drain when metrics must not run
// after enqueue. HTTP cases cover reply-on-timeout while run_impl is still
// running, and while the go task has not started. All HTTP servers in this
// binary use compute_threads=1 so the first WORKFLOW_library_init pins a
// single compute thread for the "routine not started" interleaving.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <toml/toml.hpp>

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/io/common_input.h"
#include "server/abstract_server.h"
#include "server/base_server_impl.h"
#include "server/inference_task.h"

using jinq::common::StatusCode;
using jinq::models::BaseAiModel;
using jinq::models::io_define::common_io::base64_input;
using jinq::models::io_define::common_io::byte_source;
using jinq::server::BaseAiServer;
using jinq::server::BaseAiServerImpl;
using jinq::server::InferenceResult;
using jinq::server::InferenceTask;

namespace {

struct TestOutput {
    int value = 0;
};

struct RunControl {
    std::atomic<int> runs{0};
    std::atomic<int> entered{0};
    std::atomic<bool> hold{false};
    std::atomic<bool> release{false};
    std::atomic<int> hold_from{1};
    int delay_ms = 0;
};

class SlowModel : public BaseAiModel<base64_input, TestOutput> {
public:
    explicit SlowModel(int delay_ms) : _m_delay_ms(delay_ms) {}

    explicit SlowModel(std::shared_ptr<RunControl> control)
        : _m_control(std::move(control)) {}

    StatusCode init(const toml::table&) override {
        _m_initialized = true;
        return StatusCode::OK;
    }

    StatusCode run_impl(const base64_input&, TestOutput& out) override {
        if (_m_control) {
            const int entered = _m_control->entered.fetch_add(1) + 1;
            _m_control->runs.fetch_add(1);
            if (_m_control->hold.load() &&
                entered >= _m_control->hold_from.load()) {
                while (!_m_control->release.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            } else if (_m_control->delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(_m_control->delay_ms));
            }
        } else if (_m_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(_m_delay_ms));
        }
        out.value = 1;
        return StatusCode::OK;
    }

    bool is_successfully_initialized() const override {
        return _m_initialized;
    }

private:
    int _m_delay_ms = 0;
    std::shared_ptr<RunControl> _m_control;
    bool _m_initialized = false;
};

class LifetimeServer : public BaseAiServerImpl<std::unique_ptr<SlowModel>, TestOutput> {
public:
    StatusCode init(const toml::table&) override {
        return StatusCode::SERVER_INIT_FAILED;
    }

    StatusCode init_minimal(int delay_ms) {
        auto worker = std::make_unique<SlowModel>(delay_ms);
        toml::table empty;
        if (worker->init(empty) != StatusCode::OK) {
            return StatusCode::SERVER_INIT_FAILED;
        }
        adopt_worker(std::move(worker));
        commit_workers(1);
        return StatusCode::OK;
    }

    void call_do_work(InferenceTask* req, InferenceResult* result) {
        do_work(req, result);
    }

    size_t queue_approx() const {
        return _m_workers.available_approx();
    }

protected:
    void fill_response_data(rapidjson::Document::AllocatorType& allocator,
                            rapidjson::Document& data,
                            const TestOutput& model_output,
                            const jinq::server::OutputOptions&) override {
        data.SetObject();
        data.AddMember("value", model_output.value, allocator);
    }
};

class LifetimeHttpImpl : public BaseAiServerImpl<std::unique_ptr<SlowModel>, TestOutput> {
public:
    StatusCode init(const toml::table& config) override {
        const toml::table* section_ptr = config["TEST_SERVER"].as_table();
        if (section_ptr == nullptr) {
            return StatusCode::SERVER_INIT_FAILED;
        }
        const toml::table& section = *section_ptr;
        auto common_status = parse_common_server_config(section);
        if (common_status != StatusCode::OK) {
            return common_status;
        }
        const int worker_nums = jinq::server::parse_worker_nums(section);
        if (worker_nums <= 0) {
            return StatusCode::SERVER_INIT_FAILED;
        }
        _m_control = std::make_shared<RunControl>();
        _m_control->delay_ms =
            static_cast<int>(section["fake_delay_ms"].value_or<int64_t>(0));
        for (int i = 0; i < worker_nums; ++i) {
            auto worker = std::make_unique<SlowModel>(_m_control);
            if (worker->init(config) != StatusCode::OK) {
                return StatusCode::SERVER_INIT_FAILED;
            }
            adopt_worker(std::move(worker));
        }
        if (!section.contains("server_uri")) {
            return StatusCode::SERVER_INIT_FAILED;
        }
        commit_identity(section["server_uri"].value_or<std::string>(""), "TEST", {});
        commit_workers(static_cast<size_t>(worker_nums));
        return StatusCode::OK;
    }

    size_t queue_approx() const {
        return _m_workers.available_approx();
    }

    int run_count() const {
        return _m_control ? _m_control->runs.load() : 0;
    }

    int entered() const {
        return _m_control ? _m_control->entered.load() : 0;
    }

    void hold_in_run() {
        hold_from_entered(1);
    }

    void hold_from_entered(int n) {
        if (_m_control) {
            _m_control->hold_from.store(n);
            _m_control->release.store(false);
            _m_control->hold.store(true);
        }
    }

    void release_run() {
        if (_m_control) {
            _m_control->release.store(true);
        }
    }

protected:
    void fill_response_data(rapidjson::Document::AllocatorType& allocator,
                            rapidjson::Document& data,
                            const TestOutput& model_output,
                            const jinq::server::OutputOptions&) override {
        data.SetObject();
        data.AddMember("value", model_output.value, allocator);
    }

private:
    std::shared_ptr<RunControl> _m_control;
};

class LifetimeHttpServer : public BaseAiServer {
public:
    LifetimeHttpServer() {
        _m_impl = std::make_unique<LifetimeHttpImpl>();
    }

    StatusCode init(const toml::table& config) override {
        auto status = _m_impl->init(config);
        if (status != StatusCode::OK) {
            return status;
        }
        return init_http_server(_m_impl.get());
    }

    void serve_process(WFHttpTask* task) override {
        _m_impl->serve_process(task);
    }

    bool is_successfully_initialized() const override {
        return _m_impl->is_successfully_initialized();
    }

    LifetimeHttpImpl* impl() {
        return _m_impl.get();
    }

private:
    std::unique_ptr<LifetimeHttpImpl> _m_impl;
};

bool wait_for(const std::function<bool()>& pred, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

int find_free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return 0;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return 0;
    }
    socklen_t len = sizeof(addr);
    getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    const int port = ntohs(addr.sin_port);
    close(fd);
    return port;
}

struct HttpResp {
    int status = 0;
    std::string body;
};

HttpResp send_request(int port, const std::string& path, const std::string& body) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        ADD_FAILURE() << "socket() failed";
        return {};
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) != 1 ||
        connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ADD_FAILURE() << "connect() failed";
        close(fd);
        return {};
    }

    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n";
    req << "Host: 127.0.0.1:" << port << "\r\n";
    req << "Connection: close\r\n";
    req << "Content-Type: application/json; charset=utf-8\r\n";
    req << "Authorization: Bearer test-secret\r\n";
    req << "Content-Length: " << body.size() << "\r\n\r\n";
    req << body;
    const std::string request = req.str();

    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t n = send(fd, request.data() + sent, request.size() - sent, 0);
        if (n <= 0) {
            ADD_FAILURE() << "send() failed";
            close(fd);
            return {};
        }
        sent += static_cast<size_t>(n);
    }

    timeval timeout{};
    timeout.tv_sec = 5;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string response;
    char buf[4096];
    for (;;) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<size_t>(n));
    }
    close(fd);

    HttpResp result;
    const size_t sp1 = response.find(' ');
    const size_t sp2 = response.find(' ', sp1 == std::string::npos ? 0 : sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        ADD_FAILURE() << "malformed HTTP response: " << response;
        return {};
    }
    result.status = std::stoi(response.substr(sp1 + 1, sp2 - sp1 - 1));
    const size_t header_end = response.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        result.body = response.substr(header_end + 4);
    }
    return result;
}

std::string build_config(int port, const std::string& uri, const std::string& extra) {
    std::ostringstream cfg;
    cfg << "[TEST_SERVER]\n";
    cfg << "host=\"127.0.0.1\"\n";
    cfg << "port=" << port << "\n";
    cfg << "max_connections=128\n";
    cfg << "peer_resp_timeout=5\n";
    cfg << "request_size_limit=64\n";
    cfg << "compute_threads=1\n";
    cfg << "handler_threads=4\n";
    cfg << "worker_nums=1\n";
    cfg << "max_queue_depth=0\n";
    cfg << "server_uri=\"" << uri << "\"\n";
    cfg << "auth_token=\"test-secret\"\n";
    if (extra.find("max_batch_size=") == std::string::npos) {
        cfg << "max_batch_size=1\n";
    }
    cfg << extra;
    return cfg.str();
}

struct ServerHandle {
    int port = 0;
    std::string uri;
    std::unique_ptr<LifetimeHttpServer> server;

    ServerHandle() = default;

    ServerHandle(ServerHandle&& other) noexcept
        : port(other.port), uri(std::move(other.uri)), server(std::move(other.server)) {
        other.port = 0;
    }

    ServerHandle& operator=(ServerHandle&& other) noexcept {
        if (this != &other) {
            if (server) {
                server->stop();
            }
            port = other.port;
            uri = std::move(other.uri);
            server = std::move(other.server);
            other.port = 0;
        }
        return *this;
    }

    ~ServerHandle() {
        if (server) {
            server->stop();
        }
    }
};

ServerHandle start_server(const std::string& uri, const std::string& extra) {
    ServerHandle handle;
    handle.uri = uri;
    for (int attempt = 0; attempt < 20; ++attempt) {
        const int port = find_free_port();
        if (port <= 0) {
            continue;
        }
        auto parsed = toml::parse(build_config(port, uri, extra));
        if (!parsed) {
            ADD_FAILURE() << "failed to parse test config";
            return handle;
        }
        handle.server = std::make_unique<LifetimeHttpServer>();
        const auto status = handle.server->init(std::move(parsed).table());
        if (status != StatusCode::OK) {
            ADD_FAILURE() << "test server init failed: "
                          << jinq::common::to_underlying(status);
            return handle;
        }
        if (handle.server->start("127.0.0.1", static_cast<unsigned short>(port)) == 0) {
            handle.port = port;
            return handle;
        }
        handle.server.reset();
    }
    ADD_FAILURE() << "failed to bind a test server port";
    return handle;
}

const char* k_json_body = "{\"images\":[\"aGVsbG8=\"]}";
const char* k_json_body_two = "{\"images\":[\"aGVsbG8=\",\"aGVsbG8=\"]}";

rapidjson::Document parse_json(const std::string& body) {
    rapidjson::Document doc;
    doc.Parse(body.c_str());
    return doc;
}

}  // namespace

TEST(do_work_lifetime, destructor_concurrent_with_in_flight_do_work) {
    auto impl = std::make_unique<LifetimeServer>();
    ASSERT_EQ(impl->init_minimal(200), StatusCode::OK);
    ASSERT_EQ(impl->queue_approx(), 1u);

    InferenceTask req;
    req.task_id = "lifetime";
    byte_source item;
    item.data = "x";
    req.items.push_back(std::move(item));
    InferenceResult<TestOutput> result;

    LifetimeServer* raw = impl.get();
    std::thread work([raw, &req, &result]() {
        raw->call_do_work(&req, &result);
    });

    ASSERT_TRUE(wait_for([&]() { return raw->queue_approx() == 0; }, 2000))
        << "do_work never checked out the worker";

    std::thread dtor([&impl]() {
        impl.reset();
    });

    work.join();
    dtor.join();

    EXPECT_EQ(result.model_run_status, StatusCode::OK);
    EXPECT_EQ(result.item_outputs.size(), 1u);
    EXPECT_EQ(result.item_outputs[0].value, 1);
    EXPECT_GT(result.worker_run_time_consuming, 0.0);
}

TEST(do_work_lifetime, timedgo_timeout_while_do_work_running) {
    // 100ms is too tight under ASan: the go task can start after the request
    // deadline, skip run_impl, and still return 504. Hold inside run_impl
    // so the request timer must fire while the worker is checked out.
    ServerHandle handle =
        start_server("/lifetime/timeout_running", "model_run_timeout=500\n");
    ASSERT_NE(handle.server, nullptr);
    ASSERT_GT(handle.port, 0);
    handle.server->impl()->hold_in_run();

    HttpResp resp;
    std::thread req([&]() {
        resp = send_request(handle.port, handle.uri, k_json_body);
    });

    const bool started = wait_for(
        [&]() {
            return handle.server->impl()->entered() >= 1 &&
                   handle.server->impl()->queue_approx() == 0;
        },
        5000);
    if (!started) {
        handle.server->impl()->release_run();
        req.join();
    }
    ASSERT_TRUE(started) << "run_impl never started";

    const auto t0 = std::chrono::steady_clock::now();
    req.join();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

    EXPECT_EQ(resp.status, 504);
    EXPECT_LT(elapsed_ms, 2000) << "deadline timer should answer while run_impl is held";
    EXPECT_EQ(handle.server->impl()->run_count(), 1);
    EXPECT_EQ(handle.server->impl()->queue_approx(), 0u)
        << "the in-flight run should still hold the worker after HTTP 504";
    auto doc = parse_json(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 4);
    EXPECT_FALSE(doc["partial"].GetBool());
    ASSERT_TRUE(doc["results"].IsArray());
    ASSERT_EQ(doc["results"].Size(), 1u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 4);
    EXPECT_TRUE(doc["results"][0]["data"].IsNull());

    handle.server->impl()->release_run();
    handle.server->stop();
    handle.server.reset();
}

TEST(do_work_lifetime, timedgo_timeout_before_routine_starts) {
    ServerHandle occupier =
        start_server("/lifetime/occupier", "model_run_timeout=2000\nfake_delay_ms=400\n");
    ASSERT_NE(occupier.server, nullptr);
    ASSERT_GT(occupier.port, 0);

    ServerHandle victim =
        start_server("/lifetime/victim", "model_run_timeout=100\nfake_delay_ms=400\n");
    ASSERT_NE(victim.server, nullptr);
    ASSERT_GT(victim.port, 0);

    HttpResp occupier_resp;
    std::thread occupier_req([&]() {
        occupier_resp = send_request(occupier.port, occupier.uri, k_json_body);
    });

    ASSERT_TRUE(wait_for([&]() { return occupier.server->impl()->queue_approx() == 0; }, 2000))
        << "occupier never checked out the worker / compute thread";

    const auto t0 = std::chrono::steady_clock::now();
    const auto victim_resp = send_request(victim.port, victim.uri, k_json_body);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

    EXPECT_EQ(victim_resp.status, 504);
    EXPECT_LT(elapsed_ms, 300) << "victim timer should fire while compute thread is busy";
    EXPECT_EQ(victim.server->impl()->run_count(), 0)
        << "victim run_impl must not have started";
    auto doc = parse_json(victim_resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 4);
    ASSERT_TRUE(doc["results"].IsArray());
    ASSERT_EQ(doc["results"].Size(), 1u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 4);

    occupier_req.join();
    EXPECT_EQ(occupier_resp.status, 200);
    EXPECT_EQ(occupier.server->impl()->run_count(), 1);
}

TEST(do_work_lifetime, deadline_graph_partial_keeps_prefix_success) {
    ServerHandle handle =
        start_server("/lifetime/partial", "model_run_timeout=500\n");
    ASSERT_NE(handle.server, nullptr);
    ASSERT_GT(handle.port, 0);
    handle.server->impl()->hold_from_entered(2);

    HttpResp resp;
    std::thread req([&]() {
        resp = send_request(handle.port, handle.uri, k_json_body_two);
    });

    const bool started = wait_for(
        [&]() {
            return handle.server->impl()->entered() >= 2 &&
                   handle.server->impl()->queue_approx() == 0;
        },
        5000);
    if (!started) {
        handle.server->impl()->release_run();
        req.join();
    }
    ASSERT_TRUE(started) << "second run_impl never started";

    req.join();
    EXPECT_EQ(resp.status, 200);
    auto doc = parse_json(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 68);
    EXPECT_TRUE(doc["partial"].GetBool());
    ASSERT_TRUE(doc["results"].IsArray());
    ASSERT_EQ(doc["results"].Size(), 2u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 0);
    ASSERT_TRUE(doc["results"][0]["data"].IsObject());
    EXPECT_EQ(doc["results"][0]["data"]["value"].GetInt(), 1);
    EXPECT_EQ(doc["results"][1]["status"].GetInt(), 4);
    EXPECT_TRUE(doc["results"][1]["data"].IsNull());
    EXPECT_EQ(handle.server->impl()->queue_approx(), 0u)
        << "item 1 should still hold the worker after 200 PARTIAL";

    handle.server->impl()->release_run();
    handle.server->stop();
    handle.server.reset();
}

TEST(do_work_lifetime, deadline_graph_all_success) {
    ServerHandle handle =
        start_server("/lifetime/ok", "model_run_timeout=2000\n");
    ASSERT_NE(handle.server, nullptr);
    ASSERT_GT(handle.port, 0);
    const auto resp = send_request(handle.port, handle.uri, k_json_body);
    EXPECT_EQ(resp.status, 200);
    auto doc = parse_json(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 0);
    EXPECT_FALSE(doc["partial"].GetBool());
    ASSERT_EQ(doc["results"].Size(), 1u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 0);
}

TEST(do_work_lifetime, deadline_graph_batch_replies_without_outer_timer) {
    ServerHandle handle = start_server(
        "/lifetime/batch", "model_run_timeout=500\nmax_batch_size=4\nmax_batch_delay_ms=20\n");
    ASSERT_NE(handle.server, nullptr);
    ASSERT_GT(handle.port, 0);
    const auto resp = send_request(handle.port, handle.uri, k_json_body);
    EXPECT_EQ(resp.status, 200);
    auto doc = parse_json(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 0);
    ASSERT_EQ(doc["results"].Size(), 1u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 0);
}
