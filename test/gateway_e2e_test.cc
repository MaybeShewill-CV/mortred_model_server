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
                      const std::string& body, const std::string& auth = "",
                      const std::map<std::string, std::string>& extra_headers = {}) {
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
    for (const auto& h : extra_headers) {
        req << h.first << ": " << h.second << "\r\n";
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
    // ctest inherits the operator's shell; a leftover MORTRED_API_TOKEN from
    // hand tests would otherwise make fail-closed cases start.
    ::unsetenv("MORTRED_API_TOKEN");
    ::unsetenv("MORTRED_GATEWAY_AUTH_TOKEN");
    ::unsetenv("MORTRED_METRICS_TOKEN");
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
        out << "[FAKE]\n"
            << "[FAKE_SERVER]\n"
            << "model=\"FAKE\"\n"
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
                              {"MORTRED_GATEWAY_PORT", std::to_string(gateway_port_)},
                              {"MORTRED_GATEWAY_CORS_ORIGINS",
                               "http://127.0.0.1:8787,http://localhost:8787"}});
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

TEST_F(GatewayE2ETest, prefixed_infer_matches_legacy_uri) {
    const auto r = send_request(gateway_port_, "POST", "/v1/models/FAKE/infer",
                                "{\"images\":[\"aGk=\"]}", "ext-token");
    EXPECT_EQ(r.status, 200);
    EXPECT_NE(r.body.find("\"fake\":true"), std::string::npos) << r.body;
    ASSERT_NE(r.headers.find("x-mortred-model"), r.headers.end()) << r.body;
    EXPECT_EQ(r.headers.at("x-mortred-model"), "FAKE");
}

TEST_F(GatewayE2ETest, unknown_model_id_is_404) {
    const auto r =
        send_request(gateway_port_, "POST", "/v1/models/NOPE/infer", "{}", "ext-token");
    EXPECT_EQ(r.status, 404);
    EXPECT_NE(r.body.find("\"status\":63"), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, prefixed_infer_get_is_405) {
    const auto r = send_request(gateway_port_, "GET", "/v1/models/FAKE/infer", "", "ext-token");
    EXPECT_EQ(r.status, 405);
    ASSERT_NE(r.headers.find("allow"), r.headers.end()) << r.body;
    EXPECT_EQ(r.headers.at("allow"), "POST");
}

TEST_F(GatewayE2ETest, legacy_uri_get_is_405) {
    const auto r =
        send_request(gateway_port_, "GET", "/mortred_ai_server_v1/test/fake", "", "ext-token");
    EXPECT_EQ(r.status, 405);
    ASSERT_NE(r.headers.find("allow"), r.headers.end()) << r.body;
    EXPECT_EQ(r.headers.at("allow"), "POST");
}

TEST_F(GatewayE2ETest, jobs_submit_rewrites_location_and_urls) {
    const auto r =
        send_request(gateway_port_, "POST", "/v1/models/FAKE/jobs", "{}", "ext-token");
    EXPECT_EQ(r.status, 202) << r.body;
    ASSERT_NE(r.headers.find("location"), r.headers.end()) << r.body;
    EXPECT_EQ(r.headers.at("location"), "/v1/models/FAKE/jobs/job_fake_1");
    EXPECT_NE(r.body.find("\"poll_url\":\"/v1/models/FAKE/jobs/job_fake_1\""), std::string::npos)
        << r.body;
    EXPECT_NE(r.body.find("\"result_url\":\"/v1/models/FAKE/jobs/job_fake_1/result\""),
              std::string::npos)
        << r.body;
    EXPECT_NE(r.body.find("\"upstream_path\":\"/jobs\""), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, jobs_get_forwards_to_upstream_jobs) {
    const auto r =
        send_request(gateway_port_, "GET", "/v1/models/FAKE/jobs/job_fake_1", "", "ext-token");
    EXPECT_EQ(r.status, 200) << r.body;
    EXPECT_NE(r.body.find("\"upstream_path\":\"/jobs/job_fake_1\""), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, jobs_wait_forwards_query) {
    const auto r = send_request(gateway_port_, "GET",
                                "/v1/models/FAKE/jobs/job_fake_1/wait?timeout=5", "", "ext-token");
    EXPECT_EQ(r.status, 200) << r.body;
    EXPECT_NE(r.body.find("\"upstream_path\":\"/jobs/job_fake_1/wait\""), std::string::npos)
        << r.body;
    EXPECT_NE(r.body.find("\"upstream_query\":\"timeout=5\""), std::string::npos) << r.body;
}

TEST_F(GatewayE2ETest, jobs_result_get) {
    const auto r = send_request(gateway_port_, "GET", "/v1/models/FAKE/jobs/job_fake_1/result", "",
                                "ext-token");
    EXPECT_EQ(r.status, 200) << r.body;
    EXPECT_NE(r.body.find("\"upstream_path\":\"/jobs/job_fake_1/result\""), std::string::npos)
        << r.body;
}

TEST_F(GatewayE2ETest, jobs_post_on_job_id_is_405) {
    const auto r =
        send_request(gateway_port_, "POST", "/v1/models/FAKE/jobs/job_fake_1", "{}", "ext-token");
    EXPECT_EQ(r.status, 405);
    ASSERT_NE(r.headers.find("allow"), r.headers.end()) << r.body;
    EXPECT_EQ(r.headers.at("allow"), "GET");
}

TEST_F(GatewayE2ETest, cors_preflight_allows_configured_origin) {
    const auto r =
        send_request(gateway_port_, "OPTIONS", "/mortred_ai_server_v1/test/fake", "", "",
                     {{"Origin", "http://127.0.0.1:8787"},
                      {"Access-Control-Request-Method", "POST"},
                      {"Access-Control-Request-Headers", "authorization,content-type"}});
    EXPECT_EQ(r.status, 204);
    ASSERT_NE(r.headers.find("access-control-allow-origin"), r.headers.end()) << r.body;
    EXPECT_EQ(r.headers.at("access-control-allow-origin"), "http://127.0.0.1:8787");
    ASSERT_NE(r.headers.find("access-control-allow-methods"), r.headers.end());
    EXPECT_NE(r.headers.at("access-control-allow-methods").find("POST"), std::string::npos);
}

TEST_F(GatewayE2ETest, cors_preflight_unknown_origin_has_no_allow_origin) {
    const auto r = send_request(gateway_port_, "OPTIONS", "/mortred_ai_server_v1/test/fake", "",
                                "", {{"Origin", "http://evil.example:9"}});
    EXPECT_EQ(r.status, 204);
    EXPECT_EQ(r.headers.find("access-control-allow-origin"), r.headers.end());
}

TEST_F(GatewayE2ETest, cors_headers_on_authorized_post) {
    const auto r = send_request(gateway_port_, "POST", "/mortred_ai_server_v1/test/fake",
                                "{\"images\":[\"aGk=\"]}", "ext-token",
                                {{"Origin", "http://localhost:8787"}});
    EXPECT_EQ(r.status, 200);
    ASSERT_NE(r.headers.find("access-control-allow-origin"), r.headers.end());
    EXPECT_EQ(r.headers.at("access-control-allow-origin"), "http://localhost:8787");
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

std::string keys_toml(const std::string& key, const std::string& name = "tenant-a",
                      const std::string& scope = "inference") {
    return "[keys." + name + "]\nhash = \"" +
           mortred::control::ApiKeyManager::sha256_hex(key) +
           "\"\nscope = \"" + scope + "\"\nenabled = true\n";
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
        out << "[FAKE]\n"
            << "[FAKE_SERVER]\n"
            << "model=\"FAKE\"\n"
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
                      const std::string& api_keys_content,
                      std::vector<std::pair<std::string, std::string>> extra_env = {}) {
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
        for (const auto& kv : extra_env) {
            env.push_back(kv);
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
    // deployment: it must start, and keys must be enforced. scrape token is
    // required on non-loopback so GET /metrics is not public.
    ASSERT_TRUE(StartGateway("0.0.0.0", "", keys_toml(valid_key_),
                             {{"MORTRED_METRICS_TOKEN", "scrape-token"}}));

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

TEST_F(GatewayAuthTest, AdminApiTokenIsAccepted) {
    ASSERT_TRUE(StartGateway("127.0.0.1", "", "", {{"MORTRED_API_TOKEN", "mgmt-token"}}));
    auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "mgmt-token");
    EXPECT_EQ(r.status, 200);
    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "wrong");
    EXPECT_EQ(r.status, 401);
}

TEST_F(GatewayAuthTest, AdminScopeKeyMayInfer) {
    ASSERT_TRUE(StartGateway("127.0.0.1", "", keys_toml(valid_key_, "ops", "admin")));
    const auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}", valid_key_);
    EXPECT_EQ(r.status, 200);
}

TEST_F(GatewayAuthTest, NonLoopbackWithOnlyApiTokenStarts) {
    ASSERT_TRUE(StartGateway("0.0.0.0", "", "",
                             {{"MORTRED_API_TOKEN", "mgmt-token"},
                              {"MORTRED_METRICS_TOKEN", "scrape-token"}}));
    auto r = send_request(gateway_port_, "POST", kAuthRoute, "{}");
    EXPECT_EQ(r.status, 401);
    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "mgmt-token");
    EXPECT_EQ(r.status, 200);
}

TEST_F(GatewayAuthTest, MetricsTokenUnsetStaysPublic) {
    ASSERT_TRUE(StartGateway("127.0.0.1", "ext-token", ""));
    auto r = send_request(gateway_port_, "GET", "/metrics", "");
    EXPECT_EQ(r.status, 200);
    r = send_request(gateway_port_, "GET", "/healthz", "");
    EXPECT_EQ(r.status, 200);
}

TEST_F(GatewayAuthTest, MetricsTokenRejectsAnonymousAndWrongBearer) {
    ASSERT_TRUE(StartGateway("127.0.0.1", "ext-token", "",
                             {{"MORTRED_METRICS_TOKEN", "scrape-token"}}));
    auto r = send_request(gateway_port_, "GET", "/metrics", "");
    EXPECT_EQ(r.status, 401);
    ASSERT_NE(r.headers.find("www-authenticate"), r.headers.end());
    r = send_request(gateway_port_, "GET", "/metrics", "", "wrong-scrape");
    EXPECT_EQ(r.status, 401);
    r = send_request(gateway_port_, "GET", "/metrics", "", "ext-token");
    EXPECT_EQ(r.status, 401) << "inference token must not unlock metrics scrape";
    r = send_request(gateway_port_, "GET", "/healthz", "");
    EXPECT_EQ(r.status, 200);
}

TEST_F(GatewayAuthTest, MetricsTokenAcceptsScrapeBearer) {
    ASSERT_TRUE(StartGateway("127.0.0.1", "ext-token", "",
                             {{"MORTRED_METRICS_TOKEN", "scrape-token"}}));
    auto r = send_request(gateway_port_, "GET", "/metrics", "", "scrape-token");
    EXPECT_EQ(r.status, 200);
    EXPECT_NE(r.body.find("mortred_http_requests_total"), std::string::npos);
    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "ext-token");
    EXPECT_EQ(r.status, 200);
    r = send_request(gateway_port_, "POST", kAuthRoute, "{}", "scrape-token");
    EXPECT_EQ(r.status, 401) << "metrics scrape token must not infer";
}

TEST_F(GatewayAuthTest, NonLoopbackWithoutMetricsTokenRefusesToStart) {
    EXPECT_FALSE(StartGateway("0.0.0.0", "ext-token", ""));
}

TEST_F(GatewayAuthTest, MetricsTokenMatchingInferenceRefusesToStart) {
    EXPECT_FALSE(StartGateway("127.0.0.1", "ext-token", "",
                              {{"MORTRED_METRICS_TOKEN", "ext-token"}}));
}

TEST_F(GatewayAuthTest, NonLoopbackWithDistinctMetricsTokenStarts) {
    ASSERT_TRUE(StartGateway("0.0.0.0", "ext-token", "",
                             {{"MORTRED_METRICS_TOKEN", "scrape-token"}}));
    auto r = send_request(gateway_port_, "GET", "/metrics", "");
    EXPECT_EQ(r.status, 401);
    r = send_request(gateway_port_, "GET", "/metrics", "", "scrape-token");
    EXPECT_EQ(r.status, 200);
    r = send_request(gateway_port_, "GET", "/healthz", "");
    EXPECT_EQ(r.status, 200);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
