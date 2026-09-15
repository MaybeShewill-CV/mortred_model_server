/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: gateway_multiinstance_test.cc
* Date: 26-9-6
************************************************/

// The acid test of the GatewayApp de-globalization: two gateway instances
// with different project roots, catalogs, tokens and ports serve in the SAME
// process at the same time and stay isolated. Before the refactor this was
// structurally impossible (file-scope globals); any residual shared state
// fails one of these assertions. Links control_workflow (needs workflow).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "control/api_key_manager.h"
#include "control/gateway/gateway_app.h"

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

struct HttpResp {
    int status = 0;
    std::string body;
    std::string raw;
};

HttpResp send_request(int port, const std::string& method, const std::string& path,
                      const std::string& auth = "") {
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
    req << "Content-Length: 0\r\n\r\n";
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
    HttpResp out;
    out.raw = response;
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

struct Instance {
    fs::path root;
    std::string model_id;
    std::string infer_token;
    std::string scrape_token;
    int gateway_port = 0;
    mortred::control::GatewayApp app;
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
    mortred << "[gateway]\nport=" << inst.gateway_port << "\n";
    mortred.close();
}

class GatewayMultiInstanceTest : public ::testing::Test {
  protected:
    void SetUp() override {
        base_dir_ = fs::temp_directory_path() / "mortred_gateway_multiinstance";
        std::error_code ec;
        fs::remove_all(base_dir_, ec);

        a_.root = base_dir_ / "a";
        a_.model_id = "MODEL_A";
        a_.infer_token = "token-a";
        a_.scrape_token = "scrape-a";
        b_.root = base_dir_ / "b";
        b_.model_id = "MODEL_B";
        b_.infer_token = "token-b";
        b_.scrape_token = "scrape-b";

        for (auto* inst : {&a_, &b_}) {
            inst->gateway_port = find_free_port();
            ASSERT_GT(inst->gateway_port, 0);
            // a dead model port is deliberate: a routed request must fail with
            // 503 (connection refused), which still proves catalog resolution
            write_instance_config(*inst, find_free_port());
        }

        mortred::control::GatewayInitOptions opt_a;
        opt_a.project_root = a_.root.string();
        opt_a.auth_token = a_.infer_token;
        opt_a.metrics_token = a_.scrape_token;
        opt_a.internal_token = "internal-a";
        opt_a.host = "127.0.0.1";
        opt_a.port = a_.gateway_port;
        ASSERT_TRUE(a_.app.init(opt_a)) << "instance A init failed";
        ASSERT_TRUE(a_.app.listen()) << "instance A listen failed";

        mortred::control::GatewayInitOptions opt_b;
        opt_b.project_root = b_.root.string();
        opt_b.auth_token = b_.infer_token;
        opt_b.metrics_token = b_.scrape_token;
        opt_b.internal_token = "internal-b";
        opt_b.host = "127.0.0.1";
        opt_b.port = b_.gateway_port;
        ASSERT_TRUE(b_.app.init(opt_b)) << "instance B init failed";
        ASSERT_TRUE(b_.app.listen()) << "instance B listen failed";
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

TEST_F(GatewayMultiInstanceTest, tokens_are_not_shared_between_instances) {
    // B's inference token is unknown to A and vice versa
    const auto cross_ba = send_request(a_.gateway_port, "POST",
                                       "/v1/models/" + a_.model_id + "/infer", b_.infer_token);
    EXPECT_EQ(cross_ba.status, 401);
    const auto cross_ab = send_request(b_.gateway_port, "POST",
                                       "/v1/models/" + b_.model_id + "/infer", a_.infer_token);
    EXPECT_EQ(cross_ab.status, 401);

    // each own token authenticates on its own instance (503 = routed to a dead
    // upstream, i.e. the catalog resolved and auth passed)
    const auto own_a = send_request(a_.gateway_port, "POST",
                                    "/v1/models/" + a_.model_id + "/infer", a_.infer_token);
    EXPECT_EQ(own_a.status, 503);
    const auto own_b = send_request(b_.gateway_port, "POST",
                                    "/v1/models/" + b_.model_id + "/infer", b_.infer_token);
    EXPECT_EQ(own_b.status, 503);
}

TEST_F(GatewayMultiInstanceTest, catalogs_are_not_shared_between_instances) {
    // A must not resolve B's catalog entry even with a valid A token
    const auto wrong_model =
        send_request(a_.gateway_port, "POST", "/v1/models/" + b_.model_id + "/infer",
                     a_.infer_token);
    EXPECT_EQ(wrong_model.status, 404);
}

TEST_F(GatewayMultiInstanceTest, scrape_tokens_are_not_shared_between_instances) {
    const auto foreign = send_request(a_.gateway_port, "GET", "/metrics", b_.scrape_token);
    EXPECT_EQ(foreign.status, 401);
    const auto own = send_request(a_.gateway_port, "GET", "/metrics", a_.scrape_token);
    EXPECT_EQ(own.status, 200);
}

TEST_F(GatewayMultiInstanceTest, rate_limited_key_gets_429_not_401) {
    // PR-3a regression: authenticate() used to return a null key for BOTH
    // "unknown card" and "valid card over QPS", so a throttled valid key
    // fell through to the static-token path and got 401. Now it must be
    // 429 + Retry-After. Built as a third instance whose project root
    // carries conf/api_keys.toml (a qps=2 inference key) - the gateway
    // loads it from <root>/conf on init.
    Instance c;
    c.root = base_dir_ / "c";
    c.model_id = "MODEL_C";
    c.gateway_port = find_free_port();
    ASSERT_GT(c.gateway_port, 0);
    write_instance_config(c, find_free_port());
    {
        std::ofstream keys(c.root / "conf" / "api_keys.toml");
        keys << "[keys.limited]\n"
             << "hash = \"" << mortred::control::ApiKeyManager::sha256_hex("key-c")
             << "\"\n"
             << "scope = \"inference\"\n"
             << "rate_limit_qps = 2\n";
    }
    mortred::control::GatewayInitOptions opt;
    opt.project_root = c.root.string();
    opt.metrics_token = "scrape-c";
    opt.internal_token = "internal-c";
    opt.host = "127.0.0.1";
    opt.port = c.gateway_port;
    ASSERT_TRUE(c.app.init(opt)) << "instance C init failed";
    ASSERT_TRUE(c.app.listen()) << "instance C listen failed";

    int status_503 = 0;  // auth passed, upstream dead (the multiinstance trick)
    int status_429 = 0;
    int status_401 = 0;
    std::string retry_after;
    for (int i = 0; i < 6; ++i) {
        const auto resp = send_request(c.gateway_port, "POST",
                                       "/v1/models/MODEL_C/infer", "key-c");
        if (resp.status == 503) {
            ++status_503;
        } else if (resp.status == 429) {
            ++status_429;
            const auto pos = resp.raw.find("Retry-After:");
            ASSERT_NE(pos, std::string::npos) << "429 without Retry-After header";
            retry_after = resp.raw.substr(pos + 12, 2);
        } else if (resp.status == 401) {
            ++status_401;
        }
    }
    // two admissions in the first window, the rest throttled; even with one
    // window boundary crossed mid-loop there is at most one extra 503
    EXPECT_GE(status_503, 2);
    EXPECT_LE(status_503, 3);
    EXPECT_GE(status_429, 3);
    // THE regression assertion: a valid throttled key never sees 401
    EXPECT_EQ(status_401, 0);
    // Retry-After is a whole second (the 1s fixed window rounds up)
    EXPECT_EQ(retry_after.substr(0, 1), "1");

    c.app.stop_listen();
}

TEST_F(GatewayMultiInstanceTest, healthz_stays_public_on_both_instances) {
    EXPECT_EQ(send_request(a_.gateway_port, "GET", "/healthz").status, 200);
    EXPECT_EQ(send_request(b_.gateway_port, "GET", "/healthz").status, 200);
}
