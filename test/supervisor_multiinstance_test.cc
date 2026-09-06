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
                const std::string& auth, std::string* body = nullptr) {
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
    req << "\r\n";
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
