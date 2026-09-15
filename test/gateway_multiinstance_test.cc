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
#include <vector>

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
                      const std::string& auth = "",
                      const std::vector<std::string>& extra_headers = {}) {
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
        req << h << "\r\n";
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
    int retry_after_secs = -1;
    for (int i = 0; i < 6; ++i) {
        const auto resp = send_request(c.gateway_port, "POST",
                                       "/v1/models/MODEL_C/infer", "key-c");
        if (resp.status == 503) {
            ++status_503;
        } else if (resp.status == 429) {
            ++status_429;
            // parse "Retry-After: <digits>" robustly: skip the spaces after
            // the colon, then collect the digit run
            // anchor to the header-line start so names like X-Retry-After
            // can never substring-match; every header begins after CRLF
            const auto pos = resp.raw.find("\r\nRetry-After:");
            ASSERT_NE(pos, std::string::npos) << "429 without Retry-After header";
            size_t vpos = pos + 2 + 12;  // CRLF + strlen("Retry-After:")
            while (vpos < resp.raw.size() && resp.raw[vpos] == ' ') {
                ++vpos;
            }
            std::string digits;
            while (vpos < resp.raw.size() && resp.raw[vpos] >= '0' &&
                   resp.raw[vpos] <= '9') {
                digits.push_back(resp.raw[vpos]);
                ++vpos;
            }
            ASSERT_FALSE(digits.empty()) << "Retry-After without a numeric value";
            retry_after_secs = std::atoi(digits.c_str());
        } else if (resp.status == 401) {
            ++status_401;
        }
    }
    // completeness: every request answered one of the three expected codes
    EXPECT_EQ(status_503 + status_429 + status_401, 6);
    // qps=2 admits 2 per 1-second window; at most ONE boundary can be crossed
    // by six loopback requests (~2ms each), so admissions are 2 or 4
    EXPECT_GE(status_503, 2);
    EXPECT_LE(status_503, 4);
    EXPECT_GE(status_429, 2);
    // THE regression assertion: a valid throttled key never sees 401
    EXPECT_EQ(status_401, 0);
    // Retry-After is exactly 1 (the 1..1000 ms budget rounds up to one second)
    EXPECT_EQ(retry_after_secs, 1);

    c.app.stop_listen();
}

TEST_F(GatewayMultiInstanceTest, ip_rate_limit_sheds_and_reports_retry_after) {
    // PR-3b L1: enabled limiter, burst=3 rate=3/s. All requests come from
    // 127.0.0.1 (trusted) WITHOUT XFF -> subject = the peer itself. The
    // first 3 pass L1 (then 401 at auth: no token), the rest are 429 with
    // an exact Retry-After and the RateLimit-* triple.
    Instance d;
    d.root = base_dir_ / "d";
    d.model_id = "MODEL_D";
    d.gateway_port = find_free_port();
    ASSERT_GT(d.gateway_port, 0);
    write_instance_config(d, find_free_port());
    {
        std::ofstream rl(d.root / "conf" / "mortred.toml", std::ios::app);
        rl << "\n[gateway.rate_limit]\n"
           << "enabled = true\n"
           << "rate_per_sec = 3\n"
           << "burst = 3\n"
           << "trusted_proxies = \"\"\n";  // peer is the subject (no XFF path)
    }
    mortred::control::GatewayInitOptions opt;
    opt.project_root = d.root.string();
    opt.metrics_token = "scrape-d";
    opt.auth_token = "infer-d";
    opt.internal_token = "internal-d";
    opt.host = "127.0.0.1";
    opt.port = d.gateway_port;
    ASSERT_TRUE(d.app.init(opt));
    ASSERT_TRUE(d.app.listen());

    int passed_l1 = 0;  // 401 = L1 passed, auth rejected (no token)
    int rejected_l1 = 0;
    int retry_header_seen = 0;
    int ratelimit_headers_seen = 0;
    for (int i = 0; i < 8; ++i) {
        const auto resp =
            send_request(d.gateway_port, "POST", "/v1/models/MODEL_D/infer");
        if (resp.status == 401) {
            ++passed_l1;
        } else if (resp.status == 429) {
            ++rejected_l1;
            if (resp.raw.find("\r\nRetry-After:") != std::string::npos) {
                ++retry_header_seen;
            }
            if (resp.raw.find("\r\nRateLimit-Limit:") != std::string::npos &&
                resp.raw.find("\r\nRateLimit-Remaining:") != std::string::npos &&
                resp.raw.find("\r\nRateLimit-Reset:") != std::string::npos) {
                ++ratelimit_headers_seen;
            }
        }
    }
    // burst=3 admits 3; a window flip (T=334ms, loop ~10ms) cannot happen in
    // 8 loopback requests, so exactly 3 pass
    EXPECT_EQ(passed_l1, 3);
    EXPECT_EQ(rejected_l1, 5);
    EXPECT_EQ(retry_header_seen, 5);
    EXPECT_EQ(ratelimit_headers_seen, 5);

    d.app.stop_listen();
}

TEST_F(GatewayMultiInstanceTest, xff_from_trusted_peer_buckets_by_client) {
    // trusted peer + X-Forwarded-For: the CLIENT address is the subject, so
    // three different fake clients never trip one shared bucket, while the
    // same client hammered 4 times (burst 3) trips on the 4th
    Instance d;
    d.root = base_dir_ / "d2";
    d.model_id = "MODEL_D";
    d.gateway_port = find_free_port();
    ASSERT_GT(d.gateway_port, 0);
    write_instance_config(d, find_free_port());
    {
        std::ofstream rl(d.root / "conf" / "mortred.toml", std::ios::app);
        rl << "\n[gateway.rate_limit]\n"
           << "enabled = true\n"
           << "rate_per_sec = 3\n"
           << "burst = 3\n"
           << "trusted_proxies = \"127.0.0.1\"\n";
    }
    mortred::control::GatewayInitOptions opt;
    opt.project_root = d.root.string();
    opt.metrics_token = "scrape-d";
    opt.auth_token = "infer-d";
    opt.internal_token = "internal-d";
    opt.host = "127.0.0.1";
    opt.port = d.gateway_port;
    ASSERT_TRUE(d.app.init(opt));
    ASSERT_TRUE(d.app.listen());

    // same client 4x -> 3 pass (401), 4th is 429
    int four01 = 0;
    int four29 = 0;
    for (int i = 0; i < 4; ++i) {
        const auto resp = send_request(d.gateway_port, "POST",
                                       "/v1/models/MODEL_D/infer", "",
                                       {"X-Forwarded-For: 9.9.9.9"});
        if (resp.status == 401) {
            ++four01;
        } else if (resp.status == 429) {
            ++four29;
        }
    }
    EXPECT_EQ(four01, 3);
    EXPECT_EQ(four29, 1);
    // three DIFFERENT clients, one request each: all pass (fresh buckets)
    int distinct_pass = 0;
    for (const char* client : {"1.1.1.1", "2.2.2.2", "3.3.3.3"}) {
        const std::string xff = std::string("X-Forwarded-For: ") + client;
        const auto resp =
            send_request(d.gateway_port, "POST", "/v1/models/MODEL_D/infer", "", {xff});
        if (resp.status == 401) {
            ++distinct_pass;
        }
    }
    EXPECT_EQ(distinct_pass, 3);

    d.app.stop_listen();
}

TEST_F(GatewayMultiInstanceTest, xff_from_untrusted_peer_is_ignored) {
    // THE anti-spoofing e2e: empty trusted list -> headers never honored ->
    // rotating fake XFFs still all count against the one real peer bucket
    Instance d;
    d.root = base_dir_ / "d3";
    d.model_id = "MODEL_D";
    d.gateway_port = find_free_port();
    ASSERT_GT(d.gateway_port, 0);
    write_instance_config(d, find_free_port());
    {
        std::ofstream rl(d.root / "conf" / "mortred.toml", std::ios::app);
        rl << "\n[gateway.rate_limit]\n"
           << "enabled = true\n"
           << "rate_per_sec = 3\n"
           << "burst = 3\n"
           << "trusted_proxies = \"\"\n";
    }
    mortred::control::GatewayInitOptions opt;
    opt.project_root = d.root.string();
    opt.metrics_token = "scrape-d";
    opt.auth_token = "infer-d";
    opt.internal_token = "internal-d";
    opt.host = "127.0.0.1";
    opt.port = d.gateway_port;
    ASSERT_TRUE(d.app.init(opt));
    ASSERT_TRUE(d.app.listen());

    const char* rotation[] = {"1.1.1.1", "2.2.2.2", "3.3.3.3",
                              "4.4.4.4", "5.5.5.5", "6.6.6.6"};
    int four01 = 0;
    int four29 = 0;
    for (int i = 0; i < 6; ++i) {
        const std::string xff = std::string("X-Forwarded-For: ") + rotation[i];
        const auto resp =
            send_request(d.gateway_port, "POST", "/v1/models/MODEL_D/infer", "", {xff});
        if (resp.status == 401) {
            ++four01;
        } else if (resp.status == 429) {
            ++four29;
        }
    }
    // 6 distinct SPOOFED identities, one real bucket: 3 pass, 3 rejected
    EXPECT_EQ(four01, 3);
    EXPECT_EQ(four29, 3);

    d.app.stop_listen();
}

TEST_F(GatewayMultiInstanceTest, shadow_mode_never_rejects) {
    // shadow=true: identical hammering, every request passes L1 (401 at
    // auth), the metering still runs underneath
    Instance d;
    d.root = base_dir_ / "d4";
    d.model_id = "MODEL_D";
    d.gateway_port = find_free_port();
    ASSERT_GT(d.gateway_port, 0);
    write_instance_config(d, find_free_port());
    {
        std::ofstream rl(d.root / "conf" / "mortred.toml", std::ios::app);
        rl << "\n[gateway.rate_limit]\n"
           << "shadow = true\n"
           << "rate_per_sec = 3\n"
           << "burst = 3\n"
           << "trusted_proxies = \"\"\n";
    }
    mortred::control::GatewayInitOptions opt;
    opt.project_root = d.root.string();
    opt.metrics_token = "scrape-d";
    opt.auth_token = "infer-d";
    opt.internal_token = "internal-d";
    opt.host = "127.0.0.1";
    opt.port = d.gateway_port;
    ASSERT_TRUE(d.app.init(opt));
    ASSERT_TRUE(d.app.listen());

    int four01 = 0;
    int four29 = 0;
    for (int i = 0; i < 8; ++i) {
        const auto resp =
            send_request(d.gateway_port, "POST", "/v1/models/MODEL_D/infer");
        if (resp.status == 401) {
            ++four01;
        } else if (resp.status == 429) {
            ++four29;
        }
    }
    EXPECT_EQ(four01, 8);
    EXPECT_EQ(four29, 0);

    d.app.stop_listen();
}

TEST_F(GatewayMultiInstanceTest, healthz_is_exempt_from_ip_limit) {
    Instance d;
    d.root = base_dir_ / "d5";
    d.model_id = "MODEL_D";
    d.gateway_port = find_free_port();
    ASSERT_GT(d.gateway_port, 0);
    write_instance_config(d, find_free_port());
    {
        std::ofstream rl(d.root / "conf" / "mortred.toml", std::ios::app);
        rl << "\n[gateway.rate_limit]\n"
           << "enabled = true\n"
           << "rate_per_sec = 3\n"
           << "burst = 3\n"
           << "trusted_proxies = \"\"\n";
    }
    mortred::control::GatewayInitOptions opt;
    opt.project_root = d.root.string();
    opt.metrics_token = "scrape-d";
    opt.auth_token = "infer-d";
    opt.internal_token = "internal-d";
    opt.host = "127.0.0.1";
    opt.port = d.gateway_port;
    ASSERT_TRUE(d.app.init(opt));
    ASSERT_TRUE(d.app.listen());

    int ok = 0;
    for (int i = 0; i < 12; ++i) {
        if (send_request(d.gateway_port, "GET", "/healthz").status == 200) {
            ++ok;
        }
    }
    EXPECT_EQ(ok, 12);

    d.app.stop_listen();
}

TEST_F(GatewayMultiInstanceTest, healthz_stays_public_on_both_instances) {
    EXPECT_EQ(send_request(a_.gateway_port, "GET", "/healthz").status, 200);
    EXPECT_EQ(send_request(b_.gateway_port, "GET", "/healthz").status, 200);
}
