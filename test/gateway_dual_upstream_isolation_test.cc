/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: gateway_dual_upstream_isolation_test.cc
************************************************/

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "control/ready_probe.h"

#ifndef MORTRED_FAKE_BIN_DEFAULT
#define MORTRED_FAKE_BIN_DEFAULT ""
#endif
#ifndef MORTRED_GATEWAY_BIN_DEFAULT
#define MORTRED_GATEWAY_BIN_DEFAULT ""
#endif

namespace fs = std::filesystem;
namespace {

int find_free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) { ::close(fd); return -1; }
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) { ::close(fd); return -1; }
    const int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

void stop_pid(pid_t pid) {
    if (pid <= 0) return;
    ::kill(pid, SIGTERM);
    for (int i = 0; i < 50; ++i) {
        if (::waitpid(pid, nullptr, WNOHANG) == pid) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ::kill(pid, SIGKILL);
    ::waitpid(pid, nullptr, 0);
}

pid_t spawn_bin(const char* bin, const std::vector<std::string>& args,
                const std::map<std::string, std::string>& env) {
    const pid_t pid = ::fork();
    if (pid == 0) {
        for (const auto& kv : env) ::setenv(kv.first.c_str(), kv.second.c_str(), 1);
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(bin));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        ::execv(bin, argv.data());
        ::_exit(127);
    }
    return pid;
}

struct HttpResp { int status = 0; std::string body; };

HttpResp send_request(int port, const std::string& method, const std::string& path,
                      const std::string& body, const std::string& auth) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return {};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) { ::close(fd); return {}; }
    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n";
    if (!auth.empty()) req << "Authorization: Bearer " << auth << "\r\n";
    req << "Content-Type: application/json\r\nContent-Length: " << body.size() << "\r\n\r\n" << body;
    const std::string request = req.str();
    ::send(fd, request.data(), request.size(), 0);
    std::string response;
    char buf[4096];
    timeval tv{}; tv.tv_sec = 10;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        response.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    HttpResp out; out.body = response;
    if (response.size() >= 12 && response.compare(0, 5, "HTTP/") == 0)
        out.status = std::atoi(response.c_str() + 9);
    return out;
}
}  // namespace

TEST(GatewayDualUpstreamIsolation, stop_one_keeps_other_alive) {
    const int port_a = find_free_port();
    const int port_b = find_free_port();
    const int gateway_port = find_free_port();
    ASSERT_GT(port_a, 0); ASSERT_GT(port_b, 0); ASSERT_GT(gateway_port, 0);
    ASSERT_NE(port_a, port_b);

    const fs::path root = fs::temp_directory_path() / ("mortred_dual_up_" + std::to_string(::getpid()));
    // The catalog reads conf/server (singular); the upstream entries are plain
    // server configs, so they live there too. Nothing is served from
    // conf/model in this test.
    fs::create_directories(root / "conf" / "server");
    {
        // The catalog requires exactly one [*_SERVER] section per file; bare
        // top-level keys are not a server entry and produce "routes: 0".
        std::ofstream a(root / "conf" / "server" / "model_a.toml");
        a << "[MODEL_A_SERVER]\nmodel=\"MODEL_A\"\nport=" << port_a
          << "\nserver_uri=\"/mortred_ai_server_v1/test/model_a\"\nserver_exe=\"fake_model_server.out\"\n";
        std::ofstream b(root / "conf" / "server" / "model_b.toml");
        b << "[MODEL_B_SERVER]\nmodel=\"MODEL_B\"\nport=" << port_b
          << "\nserver_uri=\"/mortred_ai_server_v1/test/model_b\"\nserver_exe=\"fake_model_server.out\"\n";
        std::ofstream mortred(root / "conf" / "mortred.toml");
        mortred << "[gateway]\nport=" << gateway_port << "\n";
    }

    const char* fake_bin = std::getenv("MORTRED_FAKE_BIN");
    if (fake_bin == nullptr) fake_bin = MORTRED_FAKE_BIN_DEFAULT;
    const char* gateway_bin = std::getenv("MORTRED_GATEWAY_BIN");
    if (gateway_bin == nullptr) gateway_bin = MORTRED_GATEWAY_BIN_DEFAULT;
    ASSERT_TRUE(fake_bin != nullptr && fake_bin[0] != '\0');
    ASSERT_TRUE(gateway_bin != nullptr && gateway_bin[0] != '\0');

    const pid_t fake_a = spawn_bin(fake_bin, {"--port", std::to_string(port_a), "--mode", "ready"}, {});
    const pid_t fake_b = spawn_bin(fake_bin, {"--port", std::to_string(port_b), "--mode", "ready"}, {});
    ASSERT_GT(fake_a, 0); ASSERT_GT(fake_b, 0);
    const pid_t gateway_pid = spawn_bin(gateway_bin, {}, {
        {"MORTRED_PROJECT_ROOT", root.string()},
        {"MORTRED_GATEWAY_AUTH_TOKEN", "ext-token"},
        {"MORTRED_API_TOKEN", "admin-token"},
        {"MORTRED_METRICS_TOKEN", "scrape-token"},
        {"MORTRED_INTERNAL_TOKEN", "int-token"},
        {"MORTRED_GATEWAY_HOST", "127.0.0.1"},
        {"MORTRED_GATEWAY_PORT", std::to_string(gateway_port)}});
    ASSERT_GT(gateway_pid, 0);

    bool ready = false;
    for (int i = 0; i < 100; ++i) {
        if (mortred::control::endpoint_ready(gateway_port, "/healthz", 500)) { ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_TRUE(ready) << "gateway did not become healthy";

    EXPECT_EQ(send_request(gateway_port, "POST", "/mortred_ai_server_v1/test/model_a", "{}", "ext-token").status, 200);
    EXPECT_EQ(send_request(gateway_port, "POST", "/mortred_ai_server_v1/test/model_b", "{}", "ext-token").status, 200);

    stop_pid(fake_a);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(send_request(gateway_port, "POST", "/mortred_ai_server_v1/test/model_a", "{}", "ext-token").status, 503);
    EXPECT_EQ(send_request(gateway_port, "POST", "/mortred_ai_server_v1/test/model_b", "{}", "ext-token").status, 200);

    stop_pid(gateway_pid);
    stop_pid(fake_b);
    std::error_code ec; fs::remove_all(root, ec);
}
