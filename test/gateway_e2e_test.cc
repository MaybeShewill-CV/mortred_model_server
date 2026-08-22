/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gateway_e2e_test.cc
* Date: 26-8-22
************************************************/

// End-to-end gateway tests: spawn the real mortred-gateway binary + a fake
// model server, then exercise routing / auth / status mapping over raw HTTP.
// Built only in full builds (needs the gateway binary + vendored workflow).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "control/ready_probe.h"

namespace fs = std::filesystem;

namespace {

#ifndef MORTRED_FAKE_BIN_DEFAULT
#define MORTRED_FAKE_BIN_DEFAULT ""
#endif
#ifndef MORTRED_GATEWAY_BIN_DEFAULT
#define MORTRED_GATEWAY_BIN_DEFAULT ""
#endif

struct HttpResp {
    int status = 0;
    std::string body;
};

HttpResp send_request(int port, const std::string& method, const std::string& path,
                      const std::string& body, const std::string& auth = "") {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return {};
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return {};
    }
    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n";
    req << "Host: 127.0.0.1\r\nConnection: close\r\n";
    if (!auth.empty()) {
        req << "Authorization: Bearer " << auth << "\r\n";
    }
    if (!body.empty()) {
        req << "Content-Length: " << body.size() << "\r\n";
    }
    req << "\r\n" << body;
    const std::string request = req.str();
    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t n = ::send(fd, request.data() + sent, request.size() - sent, 0);
        if (n <= 0) {
            ::close(fd);
            return {};
        }
        sent += static_cast<size_t>(n);
    }
    std::string response;
    char buf[4096];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);

    HttpResp out;
    const auto sp = response.find(' ');
    if (sp != std::string::npos) {
        out.status = std::atoi(response.substr(sp + 1, 3).c_str());
    }
    const auto body_pos = response.find("\r\n\r\n");
    if (body_pos != std::string::npos) {
        out.body = response.substr(body_pos + 4);
    }
    return out;
}

pid_t spawn(const std::string& exe, const std::vector<std::string>& args,
            const std::vector<std::pair<std::string, std::string>>& env) {
    const pid_t pid = ::fork();
    if (pid != 0) {
        return pid;
    }
    for (const auto& kv : env) {
        ::setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (const auto& a : args) {
        argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);
    ::execv(exe.c_str(), argv.data());
    ::_exit(127);
}

void stop(pid_t pid) {
    if (pid > 0) {
        ::kill(pid, SIGKILL);
        int status = 0;
        ::waitpid(pid, &status, 0);
    }
}

class GatewayE2ETest : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / "mortred_gateway_e2e";
        std::error_code ec;
        fs::remove_all(root_, ec);
        fs::create_directories(root_ / "conf" / "server", ec);

        model_port_ = 38500 + (::getpid() % 10000);
        gateway_port_ = model_port_ + 1;
        std::ofstream out(root_ / "conf" / "server" / "fake.toml");
        out << "[FAKE_SERVER]\n"
            << "port=" << model_port_ << "\n"
            << "server_uri=\"/mortred_ai_server_v1/test/fake\"\n"
            << "server_exe=\"fake_model_server.out\"\n";
        out.close();  // flush to disk BEFORE the gateway reads it
        std::ofstream mortred(root_ / "conf" / "mortred.toml");
        mortred << "[gateway]\nport=" << gateway_port_ << "\n";
        mortred.close();

        const char* gateway_env = std::getenv("MORTRED_GATEWAY_BIN");
        if (gateway_env == nullptr) {
            gateway_env = MORTRED_GATEWAY_BIN_DEFAULT;
        }
        ASSERT_NE(gateway_env, nullptr);
        const char* fake_env = std::getenv("MORTRED_FAKE_BIN");
        if (fake_env == nullptr) {
            fake_env = MORTRED_FAKE_BIN_DEFAULT;
        }
        ASSERT_NE(fake_env, nullptr);

        fake_pid_ = spawn(fake_env,
                          {"--port", std::to_string(model_port_), "--mode", "ready"}, {});
        gateway_pid_ = spawn(gateway_env, {},
                             {{"MORTRED_PROJECT_ROOT", root_.string()},
                              {"MORTRED_GATEWAY_AUTH_TOKEN", "ext-token"},
                              {"MORTRED_INTERNAL_TOKEN", "int-token"},
                              {"MORTRED_GATEWAY_HOST", "127.0.0.1"},
                              {"MORTRED_GATEWAY_PORT", std::to_string(gateway_port_)}});
        for (int i = 0; i < 100; ++i) {
            if (mortred::control::endpoint_ready(gateway_port_, "/healthz", 500)) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        FAIL() << "gateway did not become healthy";
    }

    void TearDown() override {
        stop(gateway_pid_);
        stop(fake_pid_);
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    fs::path root_;
    int model_port_ = 0;
    int gateway_port_ = 0;
    pid_t fake_pid_ = -1;
    pid_t gateway_pid_ = -1;
};

}  // namespace

TEST_F(GatewayE2ETest, healthz_is_public) {
    const auto r = send_request(gateway_port_, "GET", "/healthz", "");
    EXPECT_EQ(r.status, 200);
}

TEST_F(GatewayE2ETest, unknown_route_is_404) {
    const auto r = send_request(gateway_port_, "POST", "/no/such/route", "{}", "ext-token");
    EXPECT_EQ(r.status, 404);
}

TEST_F(GatewayE2ETest, model_route_requires_token) {
    const auto r =
        send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake", "{}");
    EXPECT_EQ(r.status, 401);
}

TEST_F(GatewayE2ETest, model_route_forwards_to_upstream) {
    const auto r = send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake",
                                "{\"img_data\":\"aGk=\"}", "ext-token");
    EXPECT_EQ(r.status, 200);
    EXPECT_NE(r.body.find("\"fake\":true"), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, method_not_allowed_is_405) {
    const auto r = send_request(gateway_port_, "GET", "/mortred_ai_server_v1/test/fake", "",
                                "ext-token");
    EXPECT_EQ(r.status, 405);
}

TEST_F(GatewayE2ETest, dead_upstream_maps_to_503) {
    stop(fake_pid_);
    fake_pid_ = -1;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto r = send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake", "{}",
                                "ext-token");
    EXPECT_EQ(r.status, 503);
}

TEST_F(GatewayE2ETest, metrics_endpoint_renders) {
    const auto r = send_request(gateway_port_, "GET", "/metrics", "");
    EXPECT_EQ(r.status, 200);
    EXPECT_NE(r.body.find("mortred_http_requests_total"), std::string::npos);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
