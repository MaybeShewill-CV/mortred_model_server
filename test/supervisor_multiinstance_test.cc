/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor_multiinstance_test.cc
* Date: 26-9-6
************************************************/

// The acid test of the SupervisorApp de-globalization: two supervisor
// instances with different project roots, catalogs and management tokens
// serve in the SAME process at the same time and stay isolated. Before the
// refactor this was structurally impossible (file-scope globals). Links
// control_workflow (needs workflow).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "control/supervisor/supervisor_app.h"

namespace fs = std::filesystem;

namespace {

int find_free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        ::close(fd);
        return -1;
    }
    const int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

int http_status(int port, const std::string& method, const std::string& path,
                const std::string& auth, std::string* body = nullptr,
                const std::string& req_body = "") {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return 0;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return 0;
    }
    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n";
    req << "Host: 127.0.0.1\r\nConnection: close\r\n";
    if (!auth.empty()) {
        req << "Authorization: Bearer " << auth << "\r\n";
    }
    if (!req_body.empty()) {
        req << "Content-Type: application/json\r\n";
    }
    req << "Content-Length: " << req_body.size() << "\r\n\r\n";
    req << req_body;
    const std::string request = req.str();
    ::send(fd, request.data(), request.size(), 0);
    std::string response;
    char buf[4096];
    timeval tv{};
    tv.tv_sec = 10;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    const auto sp = response.find(' ');
    if (body != nullptr) {
        const auto body_pos = response.find("\r\n\r\n");
        if (body_pos != std::string::npos) {
            *body = response.substr(body_pos + 4);
        }
    }
    return sp == std::string::npos ? 0 : std::atoi(response.substr(sp + 1, 3).c_str());
}

struct Instance {
    fs::path root;
    std::string model_id;
    std::string token;
    int port = 0;
    mortred::control::SupervisorApp app;
};

void write_instance_config(const Instance& inst, int model_port) {
    std::error_code ec;
    fs::create_directories(inst.root / "conf" / "server", ec);
    std::ofstream server(inst.root / "conf" / "server" / (inst.model_id + ".toml"));
    server << "[" << inst.model_id << "_SERVER]\n"
           << "model=\"" << inst.model_id << "\"\n"
           << "port=" << model_port << "\n"
           << "server_uri=\"/mortred_ai_server_v1/test/" << inst.model_id << "\"\n"
           << "server_exe=\"fake_model_server.out\"\n";
    server.close();
    std::ofstream mortred(inst.root / "conf" / "mortred.toml");
    mortred << "[supervisor]\n";
    mortred.close();
}

class SupervisorMultiInstanceTest : public ::testing::Test {
  protected:
    void SetUp() override {
        base_dir_ = fs::temp_directory_path() / "mortred_supervisor_multiinstance";
        std::error_code ec;
        fs::remove_all(base_dir_, ec);

        a_.root = base_dir_ / "a";
        a_.model_id = "SUPER_A";
        a_.token = "mgmt-a";
        b_.root = base_dir_ / "b";
        b_.model_id = "SUPER_B";
        b_.token = "mgmt-b";

        for (auto* inst : {&a_, &b_}) {
            inst->port = find_free_port();
            ASSERT_GT(inst->port, 0);
            write_instance_config(*inst, find_free_port());
            mortred::control::SupervisorInitOptions opt;
            opt.project_root = inst->root.string();
            opt.api_host = "127.0.0.1";
            opt.api_port = inst->port;
            opt.api_token = inst->token;
            opt.autostart_default = 0;  // supervise nothing; isolation only
            ASSERT_TRUE(inst->app.init(opt)) << "init failed for " << inst->model_id;
            ASSERT_TRUE(inst->app.listen()) << "listen failed for " << inst->model_id;
        }
    }

    void TearDown() override {
        a_.app.stop_listen();
        b_.app.stop_listen();
        std::error_code ec;
        fs::remove_all(base_dir_, ec);
    }

    fs::path base_dir_;
    Instance a_;
    Instance b_;
};

}  // namespace

TEST_F(SupervisorMultiInstanceTest, health_is_public_on_both_instances) {
    EXPECT_EQ(http_status(a_.port, "GET", "/api/v1/health", ""), 200);

    EXPECT_EQ(http_status(b_.port, "GET", "/api/v1/health", ""), 200);

}

TEST_F(SupervisorMultiInstanceTest, management_tokens_are_not_shared) {
    // B's token is unknown to A and vice versa
    EXPECT_EQ(http_status(a_.port, "GET", "/api/v1/catalog", b_.token), 401);
    EXPECT_EQ(http_status(b_.port, "GET", "/api/v1/catalog", a_.token), 401);
    // each own token works
    EXPECT_EQ(http_status(a_.port, "GET", "/api/v1/catalog", a_.token), 200);
    EXPECT_EQ(http_status(b_.port, "GET", "/api/v1/catalog", b_.token), 200);
}

TEST_F(SupervisorMultiInstanceTest, catalogs_are_not_shared) {
    std::string body;
    ASSERT_EQ(http_status(a_.port, "GET", "/api/v1/catalog", a_.token, &body), 200);
    EXPECT_NE(body.find(a_.model_id), std::string::npos) << body;
    EXPECT_EQ(body.find(b_.model_id), std::string::npos)
        << "instance A must not see instance B's catalog entries";
}

TEST_F(SupervisorMultiInstanceTest, supervisor_metrics_render_per_instance) {
    EXPECT_EQ(http_status(a_.port, "GET", "/api/v1/metrics", a_.token), 200);
    EXPECT_EQ(http_status(b_.port, "GET", "/api/v1/metrics", b_.token), 200);
}

TEST_F(SupervisorMultiInstanceTest, server_actions_require_bearer_and_post) {
    EXPECT_EQ(http_status(a_.port, "POST", "/api/v1/servers/SUPER_A/start", ""), 401);
    EXPECT_EQ(http_status(a_.port, "GET", "/api/v1/servers/SUPER_A/start", a_.token), 405);
    EXPECT_EQ(http_status(a_.port, "POST", "/api/v1/servers/nope/start", a_.token, nullptr, "{}"),
              404);
    EXPECT_EQ(http_status(a_.port, "POST", "/api/v1/servers/SUPER_A/explode", a_.token, nullptr, "{}"),
              400);
}

TEST_F(SupervisorMultiInstanceTest, start_stop_restart_return_json_ok) {
    std::string body;
    ASSERT_EQ(http_status(a_.port, "POST", "/api/v1/servers/SUPER_A/start", a_.token, &body, "{}"),
              200);
    EXPECT_NE(body.find("\"ok\""), std::string::npos) << body;

    body.clear();
    ASSERT_EQ(http_status(a_.port, "POST", "/api/v1/servers/SUPER_A/stop", a_.token, &body, "{}"),
              200);
    EXPECT_NE(body.find("\"ok\""), std::string::npos) << body;

    body.clear();
    ASSERT_EQ(http_status(a_.port, "POST", "/api/v1/servers/SUPER_A/restart", a_.token, &body, "{}"),
              200);
    EXPECT_NE(body.find("\"ok\""), std::string::npos) << body;
}

TEST_F(SupervisorMultiInstanceTest, logs_are_json_for_known_server) {
    std::string body;
    ASSERT_EQ(http_status(a_.port, "GET", "/api/v1/servers/SUPER_A/logs", a_.token, &body), 200);
    EXPECT_NE(body.find("\"lines\""), std::string::npos) << body;
    EXPECT_NE(body.find("\"offset\""), std::string::npos) << body;
    EXPECT_NE(body.find("\"total\""), std::string::npos) << body;
    EXPECT_EQ(http_status(a_.port, "POST", "/api/v1/servers/SUPER_A/logs", a_.token), 405);
    EXPECT_EQ(http_status(a_.port, "GET", "/api/v1/servers/nope/logs", a_.token), 404);
}

TEST_F(SupervisorMultiInstanceTest, graceful_restart_returns_quickly) {
    std::string body;
    const auto t0 = std::chrono::steady_clock::now();
    const int status =
        http_status(a_.port, "POST", "/api/v1/servers/SUPER_A/graceful_restart", a_.token, &body, "{}");
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    EXPECT_LT(elapsed_ms, 10000) << "drain must not wait the 120s timeout";
    EXPECT_TRUE(status == 200 || status == 500) << status << " " << body;
    EXPECT_NE(body.find("\"drained\":true"), std::string::npos) << body;
}

TEST_F(SupervisorMultiInstanceTest, keys_routes_stay_gone) {
    EXPECT_EQ(http_status(a_.port, "GET", "/api/v1/keys", a_.token), 404);
    EXPECT_EQ(http_status(a_.port, "POST", "/api/v1/keys", a_.token, nullptr, "{}"), 404);
    EXPECT_EQ(http_status(a_.port, "POST", "/api/v1/keys/reload", a_.token, nullptr, "{}"), 404);
}

TEST(SupervisorAppInit, scrape_not_required_without_gateway_binary) {
    const fs::path root =
        fs::temp_directory_path() / ("mortred_sup_init_noscrape_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "conf" / "server", ec);
    std::ofstream mortred(root / "conf" / "mortred.toml");
    mortred << "[supervisor]\n";
    mortred.close();
    std::ofstream server(root / "conf" / "server" / "ONLY.toml");
    server << "[ONLY_SERVER]\nmodel=\"ONLY\"\nport=1\nserver_uri=\"/only\"\n"
           << "server_exe=\"fake_model_server.out\"\n";
    server.close();

    mortred::control::SupervisorApp app;
    mortred::control::SupervisorInitOptions opt;
    opt.project_root = root.string();
    opt.api_host = "127.0.0.1";
    opt.api_port = find_free_port();
    opt.api_token = "mgmt-token";
    opt.autostart_default = 0;
    EXPECT_TRUE(app.init(opt));
    app.stop_listen();
    fs::remove_all(root, ec);
}

TEST(SupervisorAppInit, scrape_required_when_gateway_binary_present) {
    const fs::path root =
        fs::temp_directory_path() / ("mortred_sup_init_scrape_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "conf" / "server", ec);
    fs::create_directories(root / "_bin", ec);
    std::ofstream mortred(root / "conf" / "mortred.toml");
    mortred << "[supervisor]\n";
    mortred.close();
    std::ofstream server(root / "conf" / "server" / "ONLY.toml");
    server << "[ONLY_SERVER]\nmodel=\"ONLY\"\nport=1\nserver_uri=\"/only\"\n"
           << "server_exe=\"fake_model_server.out\"\n";
    server.close();
    std::ofstream gw(root / "_bin" / "mortred-gateway.out");
    gw << "stub\n";
    gw.close();

    mortred::control::SupervisorInitOptions opt;
    opt.project_root = root.string();
    opt.api_host = "127.0.0.1";
    opt.api_port = find_free_port();
    opt.api_token = "mgmt-token";
    opt.autostart_default = 0;

    {
        mortred::control::SupervisorApp app;
        EXPECT_FALSE(app.init(opt));
    }
    {
        mortred::control::SupervisorApp app;
        opt.metrics_token = "mgmt-token";
        opt.gateway_auth_token = "infer-token";
        EXPECT_FALSE(app.init(opt));
    }
    {
        mortred::control::SupervisorApp app;
        opt.metrics_token = "scrape-token";
        opt.gateway_auth_token = "infer-token";
        EXPECT_TRUE(app.init(opt));
        app.stop_listen();
    }
    fs::remove_all(root, ec);
}
