/************************************************
 * Author: Codex
 * File: server_e2e_contract_test.cc
 *
 * HTTP 级端到端契约测试：线程模式起真实 WFHttpServer + 假模型，
 * 验证统一 envelope、HTTP 状态码、响应头与 data:null 语义。
 * 需要 workflow 库链接（tests-only 构建未提供时自动跳过注册）。
 ************************************************/

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include "server/abstract_server.h"
#include "server/base_server_impl.h"

using jinq::common::StatusCode;
using jinq::models::BaseAiModel;
using jinq::models::io_define::common_io::base64_input;
using jinq::server::BaseAiServer;
using jinq::server::BaseAiServerImpl;

namespace {

struct TestOutput {
    int value = 0;
};

class FakeModel : public BaseAiModel<base64_input, TestOutput> {
public:
    FakeModel(int delay_ms, StatusCode fail_code)
        : _m_delay_ms(delay_ms), _m_fail_code(fail_code) {}

    StatusCode init(const toml::table&) override {
        _m_initialized = true;
        return StatusCode::OK;
    }

    StatusCode run(const base64_input&, TestOutput& out) override {
        if (_m_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(_m_delay_ms));
        }
        if (_m_fail_code != StatusCode::OK) {
            return _m_fail_code;
        }
        out.value = 1;
        return StatusCode::OK;
    }

    bool is_successfully_initialized() const override {
        return _m_initialized;
    }

private:
    int _m_delay_ms = 0;
    StatusCode _m_fail_code = StatusCode::OK;
    bool _m_initialized = false;
};

using FakeModelPtr = std::unique_ptr<FakeModel>;

class ContractTestServer : public BaseAiServer {
public:
    class Impl : public BaseAiServerImpl<FakeModelPtr, TestOutput> {
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
            const int delay_ms =
                static_cast<int>(section["fake_delay_ms"].value_or<int64_t>(0));
            const int fail_code =
                static_cast<int>(section["fake_fail_code"].value_or<int64_t>(0));
            for (int i = 0; i < worker_nums; ++i) {
                _m_working_queue.enqueue(
                    std::make_unique<FakeModel>(delay_ms, static_cast<StatusCode>(fail_code)));
            }
            if (!section.contains("server_uri")) {
                return StatusCode::SERVER_INIT_FAILED;
            }
            _m_server_uri = section["server_uri"].value_or<std::string>("");
            _m_worker_nums = static_cast<size_t>(worker_nums);
            _m_successfully_initialized = true;
            return StatusCode::OK;
        }

    protected:
        void fill_response_data(rapidjson::Document::AllocatorType& allocator,
                                rapidjson::Document& data,
                                const StatusCode& status,
                                const TestOutput& model_output) override {
            (void)status;
            data.SetObject();
            data.AddMember("ok", true, allocator);
            data.AddMember("value", model_output.value, allocator);
        }
    };

    ContractTestServer() {
        _m_impl = std::make_unique<Impl>();
    }

    ~ContractTestServer() override = default;

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

private:
    std::unique_ptr<Impl> _m_impl;
};

struct HttpResp {
    int status = 0;
    std::map<std::string, std::string> headers;
    std::string body;
};

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

HttpResp send_request(int port,
                      const std::string& method,
                      const std::string& path,
                      const std::string& body,
                      const std::vector<std::pair<std::string, std::string>>& headers) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        ADD_FAILURE() << "socket() failed";
        return HttpResp{};
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) != 1 ||
        connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ADD_FAILURE() << "connect() failed";
        close(fd);
        return HttpResp{};
    }

    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n";
    req << "Host: 127.0.0.1:" << port << "\r\n";
    req << "Connection: close\r\n";
    for (const auto& [name, value] : headers) {
        req << name << ": " << value << "\r\n";
    }
    if (!body.empty()) {
        req << "Content-Length: " << body.size() << "\r\n";
    }
    req << "\r\n" << body;
    const std::string request = req.str();

    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t n = send(fd, request.data() + sent, request.size() - sent, 0);
        // ASSERT_* 宏展开为 return;（void），不能出现在返回 HttpResp 的函数里
        if (n <= 0) {
            ADD_FAILURE() << "send() failed";
            close(fd);
            return HttpResp{};
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
    const size_t header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        ADD_FAILURE() << "malformed HTTP response: " << response;
        return HttpResp{};
    }
    const std::string header_block = response.substr(0, header_end);
    const std::string body_block = response.substr(header_end + 4);

    std::istringstream header_stream(header_block);
    std::string status_line;
    std::getline(header_stream, status_line);
    const size_t sp1 = status_line.find(' ');
    const size_t sp2 = status_line.find(' ', sp1 + 1);
    result.status = std::stoi(status_line.substr(sp1 + 1, sp2 - sp1 - 1));

    std::string line;
    while (std::getline(header_stream, line)) {
        if (line.empty() || line.back() == '\r') {
            line.pop_back();
        }
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string name = to_lower(line.substr(0, colon));
        std::string value = line.substr(colon + 1);
        while (!value.empty() && value.front() == ' ') {
            value.erase(value.begin());
        }
        result.headers[name] = value;
    }

    const auto it = result.headers.find("content-length");
    if (it != result.headers.end()) {
        const size_t len = std::stoull(it->second);
        result.body = body_block.substr(0, len);
    } else {
        result.body = body_block;
    }
    return result;
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

std::string build_config(int port, const std::string& extra, int request_size_limit_mb) {
    std::ostringstream cfg;
    cfg << "[TEST_SERVER]\n";
    cfg << "host=\"127.0.0.1\"\n";
    cfg << "port=" << port << "\n";
    cfg << "max_connections=128\n";
    cfg << "peer_resp_timeout=5\n";
    cfg << "request_size_limit=" << request_size_limit_mb << "\n";
    cfg << "compute_threads=2\n";
    cfg << "handler_threads=4\n";
    cfg << "worker_nums=1\n";
    cfg << "server_uri=\"/test/model\"\n";
    cfg << "auth_token=\"test-secret\"\n";
    // model_run_timeout 与 rate_limit_qps 不写入基底：需要覆盖它们的用例
    // 通过 extra 传入，TOML 不允许重复键（parse 会直接失败）；
    // 不传时服务端有相同默认值（500ms / 不限流）
    cfg << extra;
    return cfg.str();
}

struct ServerHandle {
    int port = 0;
    std::unique_ptr<ContractTestServer> server;

    ServerHandle() = default;

    // 用户声明的析构函数会抑制隐式移动构造/赋值；多分支 return handle
    // 无法全部走 NRVO，必须显式提供移动语义（唯一所有权转移）
    ServerHandle(ServerHandle&& other) noexcept
        : port(other.port), server(std::move(other.server)) {
        other.port = 0;
    }

    ServerHandle& operator=(ServerHandle&& other) noexcept {
        if (this != &other) {
            if (server) {
                server->stop();
            }
            port = other.port;
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

ServerHandle start_server(const std::string& extra = "",
                          int request_size_limit_mb = 64) {
    ServerHandle handle;
    for (int attempt = 0; attempt < 20; ++attempt) {
        const int port = find_free_port();
        if (port <= 0) {
            continue;
        }
        auto parsed = toml::parse(build_config(port, extra, request_size_limit_mb));
        if (!parsed) {
            ADD_FAILURE() << "failed to parse test config";
            return handle;
        }
        handle.server = std::make_unique<ContractTestServer>();
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

const std::vector<std::pair<std::string, std::string>> k_json_auth_headers = {
    {"Content-Type", "application/json; charset=utf-8"},
    {"Authorization", "Bearer test-secret"},
};

rapidjson::Document parse_body(const std::string& body) {
    rapidjson::Document doc;
    doc.Parse(body.c_str());
    return doc;
}

}  // namespace

TEST(server_e2e_contract, success_returns_200_with_envelope_and_headers) {
    ServerHandle handle = start_server();
    const std::string body =
        "{\"img_data\":\"aGVsbG8=\",\"req_id\":\"req-1\"}";
    auto resp = send_request(handle.port, "POST", "/test/model", body, k_json_auth_headers);

    EXPECT_EQ(resp.status, 200);
    EXPECT_TRUE(resp.headers["content-type"].find("application/json") != std::string::npos);
    EXPECT_EQ(resp.headers["x-request-id"], "req-1");
    EXPECT_EQ(resp.headers["cache-control"], "no-store");

    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 0);
    EXPECT_STREQ(doc["msg"].GetString(), "success");
    EXPECT_STREQ(doc["req_id"].GetString(), "req-1");
    ASSERT_TRUE(doc["data"].IsObject());
    EXPECT_TRUE(doc["data"]["ok"].GetBool());
    EXPECT_EQ(doc["data"]["value"].GetInt(), 1);
}

TEST(server_e2e_contract, bad_json_returns_400_with_null_data) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "POST", "/test/model", "not-json",
                             k_json_auth_headers);

    EXPECT_EQ(resp.status, 400);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 50);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, empty_image_returns_400_with_null_data) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"img_data\":\"\"}", k_json_auth_headers);

    EXPECT_EQ(resp.status, 400);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 3);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, missing_token_returns_401_with_www_authenticate) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"img_data\":\"aGVsbG8=\"}",
                             {{"Content-Type", "application/json"}});

    EXPECT_EQ(resp.status, 401);
    EXPECT_TRUE(resp.headers["www-authenticate"].find("Bearer") != std::string::npos);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 401);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, wrong_method_returns_405_with_allow_header) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "GET", "/test/model", "",
                             {{"Authorization", "Bearer test-secret"}});

    EXPECT_EQ(resp.status, 405);
    EXPECT_EQ(resp.headers["allow"], "POST");
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 62);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, unknown_path_returns_404) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "GET", "/nope", "",
                             {{"Authorization", "Bearer test-secret"}});

    EXPECT_EQ(resp.status, 404);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 63);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, wrong_content_type_returns_415) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"img_data\":\"aGVsbG8=\"}",
                             {{"Content-Type", "text/plain"},
                              {"Authorization", "Bearer test-secret"}});

    EXPECT_EQ(resp.status, 415);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 60);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, rate_limited_returns_429) {
    ServerHandle handle = start_server("rate_limit_qps=1\n");
    const std::string body = "{\"img_data\":\"aGVsbG8=\"}";
    auto first = send_request(handle.port, "POST", "/test/model", body, k_json_auth_headers);
    EXPECT_EQ(first.status, 200);
    auto second = send_request(handle.port, "POST", "/test/model", body, k_json_auth_headers);

    EXPECT_EQ(second.status, 429);
    auto doc = parse_body(second.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 429);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, model_timeout_returns_504) {
    ServerHandle handle = start_server("model_run_timeout=100\nfake_delay_ms=1000\n");
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"img_data\":\"aGVsbG8=\"}", k_json_auth_headers);

    EXPECT_EQ(resp.status, 504);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 4);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, model_failure_returns_500_with_null_data) {
    ServerHandle handle = start_server("fake_fail_code=2\n");
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"img_data\":\"aGVsbG8=\"}", k_json_auth_headers);

    EXPECT_EQ(resp.status, 500);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["code"].GetInt(), 2);
    EXPECT_TRUE(doc["data"].IsNull());
}

TEST(server_e2e_contract, content_length_over_limit_returns_413) {
    ServerHandle handle = start_server("", 1);
    const std::string big_body(2 * 1024 * 1024, 'a');
    auto resp = send_request(handle.port, "POST", "/test/model", big_body,
                             k_json_auth_headers);

    EXPECT_EQ(resp.status, 413);
}

TEST(server_e2e_contract, metrics_inference_duration_sum_is_positive_after_request) {
    ServerHandle handle = start_server();
    const std::string body = "{\"img_data\":\"aGVsbG8=\"}";
    auto resp = send_request(handle.port, "POST", "/test/model", body, k_json_auth_headers);
    ASSERT_EQ(resp.status, 200);

    auto metrics = send_request(handle.port, "GET", "/metrics", "", {});
    ASSERT_EQ(metrics.status, 200);
    const std::string key = "mortred_inference_duration_ms_sum";
    const auto pos = metrics.body.find(key);
    ASSERT_NE(pos, std::string::npos) << metrics.body;
    const auto value_pos = metrics.body.find(' ', pos + key.size());
    ASSERT_NE(value_pos, std::string::npos);
    const double sum = std::atof(
        metrics.body.substr(value_pos + 1, metrics.body.find('\n', value_pos)).c_str());
    EXPECT_GT(sum, 0.0) << "inference duration histogram must observe the real "
                        << "run time, not the pre-assignment zero";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
