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
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "control/api_key_manager.h"
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
    std::map<std::string, std::string> headers;
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
    const auto headers_end = response.find("\r\n\r\n");
    if (headers_end != std::string::npos) {
        size_t line_start = response.find("\r\n") + 2;  // skip the status line
        while (line_start < headers_end) {
            const size_t line_end = response.find("\r\n", line_start);
            const std::string line =
                response.substr(line_start, line_end - line_start);
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                std::string name = line.substr(0, colon);
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                std::string value = line.substr(colon + 1);
                if (!value.empty() && value[0] == ' ') {
                    value.erase(0, 1);
                }
                out.headers[name] = value;
            }
            line_start = line_end + 2;
        }
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
    EXPECT_NE(r.body.find("\"status\":63"), std::string::npos) << r.body;
    EXPECT_NE(r.body.find("\"errors\""), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, model_route_requires_token) {
    const auto r =
        send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake", "{}");
    EXPECT_EQ(r.status, 401);
    EXPECT_NE(r.body.find("\"status\":401"), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, model_route_forwards_to_upstream) {
    const auto r = send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake",
                                "{\"images\":[\"aGk=\"]}", "ext-token");
    EXPECT_EQ(r.status, 200);
    EXPECT_NE(r.body.find("\"fake\":true"), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, method_not_allowed_is_405) {
    const auto r = send_request(gateway_port_, "GET", "/mortred_ai_server_v1/test/fake", "",
                                "ext-token");
    EXPECT_EQ(r.status, 405);
    EXPECT_NE(r.body.find("\"status\":62"), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, dead_upstream_maps_to_503) {
    stop(fake_pid_);
    fake_pid_ = -1;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto r = send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake", "{}",
                                "ext-token");
    EXPECT_EQ(r.status, 503);
    EXPECT_NE(r.body.find("\"status\":65"), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, upstream_overload_headers_pass_through) {
    // restart the fake in overloaded mode: 429 + Retry-After must arrive at
    // the client verbatim (the whole point of end-to-end backpressure)
    stop(fake_pid_);
    const char* fake_bin = std::getenv("MORTRED_FAKE_BIN");
    if (fake_bin == nullptr) {
        fake_bin = MORTRED_FAKE_BIN_DEFAULT;
    }
    fake_pid_ = spawn(fake_bin, {"--port", std::to_string(model_port_), "--mode", "overloaded"},
                      {});
    for (int i = 0; i < 50; ++i) {
        if (mortred::control::endpoint_ready(model_port_, "/ready", 500)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const auto r = send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake", "{}",
                                "ext-token");
    EXPECT_EQ(r.status, 429);
    ASSERT_NE(r.headers.find("retry-after"), r.headers.end())
        << "gateway must forward the upstream Retry-After hint";
    EXPECT_EQ(r.headers.at("retry-after"), "2");
}

TEST_F(GatewayE2ETest, metrics_endpoint_renders) {
    const auto r = send_request(gateway_port_, "GET", "/metrics", "");
    EXPECT_EQ(r.status, 200);
    EXPECT_NE(r.body.find("mortred_http_requests_total"), std::string::npos);
}

// ---- auth-mode e2e coverage -----------------------------------------------
// Regression suite for the gateway auth fail-open: the legacy static-token
// fallback must never let an EMPTY token authorize requests when API keys are
// the configured auth, and startup must be fail-closed for misconfigurations.

namespace {

const char kAuthRoute[] = "/mortred_ai_server_v1/test/fake";

std::string keys_toml(const std::string& key, const std::string& name = "tenant-a") {
    return "[keys." + name + "]\nhash = \"" +
           mortred::control::ApiKeyManager::sha256_hex(key) +
           "\"\nscope = \"inference\"\nenabled = true\n";
}

}  // namespace

class GatewayAuthTest : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / "mortred_gateway_auth_e2e";
        std::error_code ec;
        fs::remove_all(root_, ec);
        fs::create_directories(root_ / "conf" / "server", ec);

        // distinct port pool from GatewayE2ETest (38500 + pid % 10000): the
        // same binary runs both fixtures, so use a fixed non-overlapping range
        model_port_ = 49000 + seq_++ * 2;
        gateway_port_ = model_port_ + 1;
        std::ofstream out(root_ / "conf" / "server" / "fake.toml");
        out << "[FAKE_SERVER]\n"
            << "port=" << model_port_ << "\n"
            << "server_uri=\"" << kAuthRoute << "\"\n"
            << "server_exe=\"fake_model_server.out\"\n";
        out.close();
        std::ofstream mortred(root_ / "conf" / "mortred.toml");
        mortred << "[gateway]\nport=" << gateway_port_ << "\n";
        mortred.close();

        fake_pid_ = spawn(env_or_default("MORTRED_FAKE_BIN", MORTRED_FAKE_BIN_DEFAULT),
                          {"--port", std::to_string(model_port_), "--mode", "ready"}, {});
    }

    void TearDown() override {
        stop(gateway_pid_);
        stop(fake_pid_);
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    static std::string env_or_default(const char* name, const char* fallback) {
        const char* v = std::getenv(name);
        return v == nullptr ? std::string(fallback) : std::string(v);
    }

    /*** starts the gateway with the given auth configuration. Returns true
     * only if the process stayed up and became healthy; a refused start
     * (fail-closed) or a crash reports false and reaps the child. */
    bool StartGateway(const std::string& host, const std::string& auth_token,
                      const std::string& api_keys_content) {
        if (!api_keys_content.empty()) {
            std::ofstream keys(root_ / "conf" / "api_keys.toml");
            keys << api_keys_content;
            keys.close();
        }
        const std::string gateway_bin =
            env_or_default("MORTRED_GATEWAY_BIN", MORTRED_GATEWAY_BIN_DEFAULT);
        if (gateway_bin.empty()) {
            ADD_FAILURE() << "no gateway binary available";
            return false;
        }
        std::vector<std::pair<std::string, std::string>> env = {
            {"MORTRED_PROJECT_ROOT", root_.string()},
            {"MORTRED_GATEWAY_HOST", host},
            {"MORTRED_GATEWAY_PORT", std::to_string(gateway_port_)},
            {"MORTRED_INTERNAL_TOKEN", "int-token"},
        };
        if (!auth_token.empty()) {
            env.emplace_back("MORTRED_GATEWAY_AUTH_TOKEN", auth_token);
        }
        gateway_pid_ = spawn(gateway_bin, {}, env);
        for (int i = 0; i < 100; ++i) {
            int status = 0;
            const pid_t reaped = ::waitpid(gateway_pid_, &status, WNOHANG);
            if (reaped == gateway_pid_) {
                gateway_pid_ = -1;  // exited during startup: refused
                return false;
            }
            if (mortred::control::endpoint_ready(gateway_port_, "/healthz", 500)) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return false;
    }

    fs::path root_;
    int model_port_ = 0;
    int gateway_port_ = 0;
    pid_t fake_pid_ = -1;
    pid_t gateway_pid_ = -1;
    const std::string valid_key_ = "valid-api-key-0001";
    static int seq_;
};

int GatewayAuthTest::seq_ = 0;

TEST_F(GatewayAuthTest, KeysOnlyLoopbackRejectsAnonymousRequests) {
    // regression for the fail-open: with api_keys configured and NO static
    // token, an empty-token fallback must NOT authorize anonymous requests
    ASSERT_TRUE(StartGateway("127.0.0.1", "", keys_toml(valid_key_)));

    auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}");
    EXPECT_EQ(r.status, 401) << "anonymous request must be denied with keys-only auth";

    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "forged-token");
    EXPECT_EQ(r.status, 401) << "forged token must be denied with keys-only auth";
}

TEST_F(GatewayAuthTest, KeysOnlyLoopbackAcceptsValidKey) {
    ASSERT_TRUE(StartGateway("127.0.0.1", "", keys_toml(valid_key_)));
    const auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}", valid_key_);
    EXPECT_EQ(r.status, 200);
}

TEST_F(GatewayAuthTest, StaticTokenFallbackStillWorksWhenConfigured) {
    // legacy compatibility preserved: with BOTH auth mechanisms configured, a
    // request carrying the static token passes even when the key lookup fails
    ASSERT_TRUE(StartGateway("127.0.0.1", "ext-token", keys_toml(valid_key_)));

    auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "ext-token");
    EXPECT_EQ(r.status, 200) << "static token fallback must survive with keys configured";

    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", valid_key_);
    EXPECT_EQ(r.status, 200);

    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "totally-wrong");
    EXPECT_EQ(r.status, 401);
}

TEST_F(GatewayAuthTest, NonLoopbackKeysOnlyStartsAndEnforcesKeys) {
    // a non-loopback listener with API keys (no static token) is a legitimate
    // deployment: it must start, and keys must be enforced
    ASSERT_TRUE(StartGateway("0.0.0.0", "", keys_toml(valid_key_)));

    auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}");
    EXPECT_EQ(r.status, 401);

    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", valid_key_);
    EXPECT_EQ(r.status, 200);
}

TEST_F(GatewayAuthTest, NonLoopbackWithoutAnyAuthRefusesToStart) {
    // fail-closed preserved: no auth mechanism at all on a public listener
    EXPECT_FALSE(StartGateway("0.0.0.0", "", ""));
}

TEST_F(GatewayAuthTest, BrokenApiKeysWithoutTokenRefusesToStart) {
    // an operator who configured keys wants authentication; a broken file
    // must not silently downgrade to an unauthenticated listener
    EXPECT_FALSE(StartGateway("127.0.0.1", "", "this is not [valid toml"));
}

TEST_F(GatewayAuthTest, BrokenApiKeysWithTokenFallsBackToStaticToken) {
    // with a static token configured the gateway may start, but the parse
    // failure must be loud and only the token must authenticate
    ASSERT_TRUE(StartGateway("127.0.0.1", "ext-token", "this is not [valid toml"));

    auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}");
    EXPECT_EQ(r.status, 401);

    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "ext-token");
    EXPECT_EQ(r.status, 200);

    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", valid_key_);
    EXPECT_EQ(r.status, 401) << "unparsed key file must not authenticate anything";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
