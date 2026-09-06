/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: supervisor_malformed_e2e_test.cc
* Date: 26-9-6
************************************************/

// The supervisor management listener must survive malformed request lines.
// Workflow leaves get_method() null when the request line cannot be parsed;
// the supervisor process() used to construct std::string from that null
// pointer (UB, typically a crash). This test drives garbage bytes and an
// empty-method request line into a real mortred-supervisor.out and asserts
// the process stays alive and keeps serving valid requests afterwards.
// Full builds only (needs the supervisor binary).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

#ifndef MORTRED_SUPERVISOR_BIN_DEFAULT
#define MORTRED_SUPERVISOR_BIN_DEFAULT ""
#endif

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

// send raw bytes, read whatever comes back (possibly nothing) within a bounded
// time; the exact reply to garbage is workflow's business, not the contract
std::string raw_roundtrip(int port, const std::string& bytes, int timeout_sec = 3) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return "";
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return "";
    }
    size_t sent = 0;
    while (sent < bytes.size()) {
        const ssize_t n = ::send(fd, bytes.data() + sent, bytes.size() - sent, 0);
        if (n <= 0) {
            ::close(fd);
            return "";
        }
        sent += static_cast<size_t>(n);
    }
    timeval tv{};
    tv.tv_sec = timeout_sec;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    std::string response;
    char buf[1024];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    return response;
}

int status_of(const std::string& response) {
    const auto sp = response.find(' ');
    if (sp == std::string::npos) {
        return 0;
    }
    return std::atoi(response.substr(sp + 1, 3).c_str());
}

pid_t spawn_supervisor(const std::string& exe, const fs::path& root, int port) {
    const pid_t pid = ::fork();
    if (pid != 0) {
        return pid;
    }
    ::unsetenv("MORTRED_AUTH_TOKEN");
    ::unsetenv("MORTRED_GATEWAY_AUTH_TOKEN");
    ::unsetenv("MORTRED_METRICS_TOKEN");
    ::unsetenv("MORTRED_INTERNAL_TOKEN");
    const std::vector<std::pair<const char*, std::string>> env = {
        {"MORTRED_PROJECT_ROOT", root.string()},
        {"MORTRED_CONTROL_CONFIG", (root / "conf" / "mortred.toml").string()},
        {"MORTRED_API_HOST", "127.0.0.1"},
        {"MORTRED_API_PORT", std::to_string(port)},
        {"MORTRED_API_TOKEN", "mgmt-secret"},
        {"MORTRED_AUTOSTART", "false"}};
    for (const auto& kv : env) {
        ::setenv(kv.first, kv.second.c_str(), 1);
    }
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
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

class SupervisorMalformedE2ETest : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / "mortred_supervisor_malformed_e2e";
        std::error_code ec;
        fs::remove_all(root_, ec);
        fs::create_directories(root_ / "conf" / "server", ec);
        std::ofstream mortred(root_ / "conf" / "mortred.toml");
        mortred << "[supervisor]\n";
        mortred.close();

        const char* env = std::getenv("MORTRED_SUPERVISOR_BIN");
        if (env == nullptr || *env == '\0') {
            env = MORTRED_SUPERVISOR_BIN_DEFAULT;
        }
        ASSERT_NE(env, nullptr);
        ASSERT_STRNE(env, "") << "MORTRED_SUPERVISOR_BIN must point at mortred-supervisor.out";

        port_ = find_free_port();
        ASSERT_GT(port_, 0);
        pid_ = spawn_supervisor(env, root_, port_);
        for (int i = 0; i < 100; ++i) {
            const auto resp =
                raw_roundtrip(port_, "GET /api/v1/health HTTP/1.1\r\nHost: t\r\n"
                                     "Connection: close\r\n\r\n");
            if (status_of(resp) == 200) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        FAIL() << "supervisor did not become healthy";
    }

    void TearDown() override {
        stop(pid_);
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    fs::path root_;
    int port_ = 0;
    pid_t pid_ = -1;
};

}  // namespace

TEST_F(SupervisorMalformedE2ETest, empty_method_request_line_does_not_kill_the_supervisor) {
    // request line with an empty method: workflow hands process() a null
    // get_method(); the supervisor must reply (or close) and stay alive
    const auto resp = raw_roundtrip(
        port_, " /api/v1/health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    (void)resp;  // any reply / clean close is acceptable
    EXPECT_EQ(::kill(pid_, 0), 0) << "supervisor died on an empty-method request line";

    // binary garbage must not take the process down either
    (void)raw_roundtrip(port_, std::string("\x00\x01\x02 garbage \r\n\r\n", 17));
    EXPECT_EQ(::kill(pid_, 0), 0) << "supervisor died on binary garbage";

    // the survivorship contract: valid management requests keep working
    const auto health = raw_roundtrip(
        port_, "GET /api/v1/health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    EXPECT_EQ(status_of(health), 200) << health;

    const auto catalog = raw_roundtrip(
        port_, "GET /api/v1/catalog HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
               "Authorization: Bearer mgmt-secret\r\n\r\n");
    EXPECT_EQ(status_of(catalog), 200) << catalog;
}
