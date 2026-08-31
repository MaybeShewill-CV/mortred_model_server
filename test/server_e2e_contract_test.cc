/************************************************
 * Author: Codex
 * File: server_e2e_contract_test.cc
 *
 * HTTP-level end-to-end contract test: runs a real WFHttpServer + fake model
 * in thread mode, verifying the unified envelope, HTTP status codes, response
 * headers, and data:null semantics. Requires workflow linkage (auto-skips when
 * the tests-only build does not provide it).
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

    StatusCode run_impl(const base64_input& in, TestOutput& out) override {
        if (_m_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(_m_delay_ms));
        }
        if (_m_fail_code != StatusCode::OK) {
            return _m_fail_code;
        }
        // per-item failure trigger for the batch isolation test: a payload
        // containing "fail-me" fails THIS item only (default run_batch loops
        // run_impl, so the batch mates keep their results)
        if (in.input_image_content.find("fail-me") != std::string::npos) {
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        // echo the payload length: the batch distribution test detects any
        // cross-entry result mixup through per-request distinct values
        out.value = static_cast<int>(in.input_image_content.size());
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
                // lifecycle contract: workers must be initialized before serving;
                // the old test relied on run() not checking initialization
                auto worker = std::make_unique<FakeModel>(delay_ms, static_cast<StatusCode>(fail_code));
                if (worker->init(config) != StatusCode::OK) {
                    return StatusCode::SERVER_INIT_FAILED;
                }
                _m_working_queue.enqueue(std::move(worker));
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
                                const TestOutput& model_output,
                                const jinq::server::OutputOptions& options) override {
            (void)options;
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
        // ASSERT_* expands to return; (void), so it cannot appear in a function
        // returning HttpResp
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
    // model_run_timeout and rate_limit_qps are not written to the base config:
    // cases that need them pass them via extra, because TOML forbids duplicate
    // keys (parse would fail outright); when absent the server uses the same
    // defaults (500ms / unlimited)
    cfg << extra;
    return cfg.str();
}

struct ServerHandle {
    int port = 0;
    std::unique_ptr<ContractTestServer> server;

    ServerHandle() = default;

    // a user-declared destructor suppresses implicit move ctor/assignment;
    // multi-branch return handles cannot all use NRVO, so move semantics must
    // be provided explicitly (unique ownership transfer)
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
        "{\"images\":[\"aGVsbG8=\"],\"req_id\":\"req-1\"}";
    auto resp = send_request(handle.port, "POST", "/test/model", body, k_json_auth_headers);

    EXPECT_EQ(resp.status, 200);
    EXPECT_TRUE(resp.headers["content-type"].find("application/json") != std::string::npos);
    EXPECT_EQ(resp.headers["x-request-id"], "req-1");
    EXPECT_EQ(resp.headers["cache-control"], "no-store");

    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 0);
    EXPECT_STREQ(doc["status_str"].GetString(), "OK");
    EXPECT_STREQ(doc["task_id"].GetString(), "req-1");
    ASSERT_TRUE(doc["model"].IsObject());
    EXPECT_TRUE(doc["results"].IsArray());
    ASSERT_EQ(doc["results"].Size(), 1u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 0);
    ASSERT_TRUE(doc["results"][0]["data"].IsObject());
    EXPECT_TRUE(doc["results"][0]["data"]["ok"].GetBool());
    EXPECT_EQ(doc["results"][0]["data"]["value"].GetInt(), 8);
    EXPECT_FALSE(doc["partial"].GetBool());
    EXPECT_TRUE(doc["server_time_ms"].IsNumber());
}

TEST(server_e2e_contract, bad_json_returns_400_with_pointer_error) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "POST", "/test/model", "not-json",
                             k_json_auth_headers);

    EXPECT_EQ(resp.status, 400);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 50);
    ASSERT_TRUE(doc["errors"].IsArray());
    ASSERT_GT(doc["errors"].Size(), 0u);
    EXPECT_STREQ(doc["errors"][0]["pointer"].GetString(), "/");
    EXPECT_TRUE(doc["results"].IsArray());
}

TEST(server_e2e_contract, legacy_img_data_returns_422_with_migration_hint) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"img_data\":\"aGVsbG8=\"}", k_json_auth_headers);

    EXPECT_EQ(resp.status, 422);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 66);
    ASSERT_EQ(doc["errors"].Size(), 1u);
    EXPECT_STREQ(doc["errors"][0]["pointer"].GetString(), "/img_data");
    EXPECT_NE(std::string(doc["errors"][0]["message"].GetString()).find("images"),
              std::string::npos)
        << "the rejection must point the legacy client at images[]";

    // never succeeds, even alongside a valid envelope
    auto both = send_request(handle.port, "POST", "/test/model",
                             "{\"img_data\":\"aGVsbG8=\",\"images\":[\"aGVsbG8=\"]}",
                             k_json_auth_headers);
    EXPECT_EQ(both.status, 422);
}

TEST(server_e2e_contract, strict_envelope_rejects_unknown_fields_and_bad_params) {
    ServerHandle handle = start_server();

    auto unknown = send_request(handle.port, "POST", "/test/model",
                                "{\"images\":[\"aGVsbG8=\"],\"foo\":1}", k_json_auth_headers);
    EXPECT_EQ(unknown.status, 422);
    auto unknown_doc = parse_body(unknown.body);
    ASSERT_EQ(unknown_doc["errors"].Size(), 1u);
    EXPECT_STREQ(unknown_doc["errors"][0]["pointer"].GetString(), "/foo");

    // this server declares no param specs: any params key is unknown
    auto params = send_request(
        handle.port, "POST", "/test/model",
        "{\"images\":[\"aGVsbG8=\"],\"params\":{\"score_threshold\":0.5}}", k_json_auth_headers);
    EXPECT_EQ(params.status, 422);
    auto params_doc = parse_body(params.body);
    ASSERT_EQ(params_doc["errors"].Size(), 1u);
    EXPECT_STREQ(params_doc["errors"][0]["pointer"].GetString(), "/params/score_threshold");
}

TEST(server_e2e_contract, multi_image_results_align_with_items) {
    ServerHandle handle = start_server();
    const std::string short_payload(8, 'a');
    const std::string long_payload(21, 'b');
    auto resp = send_request(
        handle.port, "POST", "/test/model",
        "{\"images\":[\"" + short_payload + "\",\"" + long_payload +
            "\"],\"req_id\":\"multi\"}",
        k_json_auth_headers);

    ASSERT_EQ(resp.status, 200);
    auto doc = parse_body(resp.body);
    ASSERT_EQ(doc["results"].Size(), 2u);
    EXPECT_EQ(doc["results"][0]["data"]["value"].GetInt(), 8);
    EXPECT_EQ(doc["results"][1]["data"]["value"].GetInt(), 21);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 0);
    EXPECT_EQ(doc["results"][1]["status"].GetInt(), 0);
}

TEST(server_e2e_contract, multi_image_failure_is_isolated_per_item) {
    ServerHandle handle = start_server();
    auto resp = send_request(
        handle.port, "POST", "/test/model",
        "{\"images\":[\"aaaaaaaa\",\"fail-me\",\"ccccccccccccccccccccccccccc\"],"
        "\"req_id\":\"iso\"}",
        k_json_auth_headers);

    // one non-timeout item failure surfaces at the HTTP layer (aggregate =
    // first error), while every item still reports its own status below
    ASSERT_EQ(resp.status, 500);
    auto doc = parse_body(resp.body);
    EXPECT_EQ(doc["status"].GetInt(), 2);
    ASSERT_EQ(doc["results"].Size(), 3u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 0);
    EXPECT_EQ(doc["results"][0]["data"]["value"].GetInt(), 8);
    EXPECT_EQ(doc["results"][1]["status"].GetInt(), 2);
    EXPECT_TRUE(doc["results"][1]["data"].IsNull());
    EXPECT_EQ(doc["results"][2]["status"].GetInt(), 0);
    EXPECT_EQ(doc["results"][2]["data"]["value"].GetInt(), 27);
}

TEST(server_e2e_contract, request_item_limit_returns_413) {
    ServerHandle handle = start_server("max_request_items=2\n");
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"images\":[\"a\",\"b\",\"c\"]}", k_json_auth_headers);
    EXPECT_EQ(resp.status, 413);
    auto doc = parse_body(resp.body);
    EXPECT_EQ(doc["status"].GetInt(), 67);
    ASSERT_EQ(doc["errors"].Size(), 1u);
    EXPECT_STREQ(doc["errors"][0]["pointer"].GetString(), "/images");
}

TEST(server_e2e_contract, raw_body_encoding_returns_the_same_envelope) {
    ServerHandle handle = start_server();
    const std::string payload(23, 'r');
    auto resp = send_request(handle.port, "POST", "/test/model", payload,
                             {{"Content-Type", "image/png"},
                              {"Authorization", "Bearer test-secret"},
                              {"X-Request-ID", "raw-1"}});
    ASSERT_EQ(resp.status, 200);
    EXPECT_EQ(resp.headers["x-request-id"], "raw-1");
    auto doc = parse_body(resp.body);
    EXPECT_EQ(doc["task_id"].GetString(), std::string("raw-1"));
    ASSERT_EQ(doc["results"].Size(), 1u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 0);
    EXPECT_EQ(doc["results"][0]["data"]["value"].GetInt(), 23);

    // octet-stream is accepted too; the type is informational
    auto octet = send_request(handle.port, "POST", "/test/model", payload,
                              {{"Content-Type", "application/octet-stream"},
                               {"Authorization", "Bearer test-secret"}});
    EXPECT_EQ(octet.status, 200);
}

TEST(server_e2e_contract, raw_and_json_encodings_produce_identical_results) {
    ServerHandle handle = start_server();
    const std::string payload(31, 'x');
    const std::string b64 = "eHh4";  // "xxx" - content only matters by length here
    (void)b64;

    auto raw = send_request(handle.port, "POST", "/test/model", payload,
                            {{"Content-Type", "image/jpeg"},
                             {"Authorization", "Bearer test-secret"},
                             {"X-Request-ID", "same"}});
    auto json = send_request(handle.port, "POST", "/test/model",
                             "{\"images\":[\"" + std::string(31, 'x') + "\"],\"req_id\":\"same\"}",
                             k_json_auth_headers);
    ASSERT_EQ(raw.status, 200);
    ASSERT_EQ(json.status, 200);

    auto raw_doc = parse_body(raw.body);
    auto json_doc = parse_body(json.body);
    // encoding must not change semantics: same task echo, same item results
    EXPECT_EQ(raw_doc["task_id"].GetString(), std::string("same"));
    ASSERT_EQ(raw_doc["results"].Size(), json_doc["results"].Size());
    EXPECT_EQ(raw_doc["results"][0]["status"].GetInt(), json_doc["results"][0]["status"].GetInt());
    EXPECT_EQ(raw_doc["results"][0]["data"]["value"].GetInt(),
              json_doc["results"][0]["data"]["value"].GetInt());
    EXPECT_EQ(raw_doc["partial"].GetBool(), json_doc["partial"].GetBool());
}

TEST(server_e2e_contract, raw_headers_reject_with_the_same_422_semantics) {
    ServerHandle handle = start_server();
    const std::string payload(8, 'y');

    // this test server declares no params: the header param is unknown
    auto unknown = send_request(handle.port, "POST", "/test/model", payload,
                                {{"Content-Type", "image/png"},
                                 {"Authorization", "Bearer test-secret"},
                                 {"X-Mortred-Params", R"({"score_threshold":0.5})"}});
    ASSERT_EQ(unknown.status, 422);
    auto doc = parse_body(unknown.body);
    ASSERT_EQ(doc["errors"].Size(), 1u);
    EXPECT_STREQ(doc["errors"][0]["pointer"].GetString(), "/params/score_threshold");

    auto malformed = send_request(handle.port, "POST", "/test/model", payload,
                                  {{"Content-Type", "image/png"},
                                   {"Authorization", "Bearer test-secret"},
                                   {"X-Mortred-Params", "not-json"}});
    ASSERT_EQ(malformed.status, 422);
    auto bad_doc = parse_body(malformed.body);
    EXPECT_STREQ(bad_doc["errors"][0]["pointer"].GetString(), "/params");

    // empty raw body -> 400 with /body pointer
    auto empty = send_request(handle.port, "POST", "/test/model", "",
                              {{"Content-Type", "image/png"},
                               {"Authorization", "Bearer test-secret"}});
    EXPECT_EQ(empty.status, 400);
    auto empty_doc = parse_body(empty.body);
    EXPECT_STREQ(empty_doc["errors"][0]["pointer"].GetString(), "/body");

    // unsupported types keep the 415
    auto bad_type = send_request(handle.port, "POST", "/test/model", payload,
                                 {{"Content-Type", "video/mp4"},
                                  {"Authorization", "Bearer test-secret"}});
    EXPECT_EQ(bad_type.status, 415);
}

TEST(server_e2e_contract, missing_token_returns_401_with_www_authenticate) {
    ServerHandle handle = start_server();
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"images\":[\"aGVsbG8=\"]}",
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
                             "{\"images\":[\"aGVsbG8=\"]}",
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
    const std::string body = "{\"images\":[\"aGVsbG8=\"]}";
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
                             "{\"images\":[\"aGVsbG8=\"]}", k_json_auth_headers);

    EXPECT_EQ(resp.status, 504);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 4);
    EXPECT_TRUE(doc["results"].IsArray());
}

TEST(server_e2e_contract, model_failure_returns_500_with_null_data) {
    ServerHandle handle = start_server("fake_fail_code=2\n");
    auto resp = send_request(handle.port, "POST", "/test/model",
                             "{\"images\":[\"aGVsbG8=\"]}", k_json_auth_headers);

    EXPECT_EQ(resp.status, 500);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_EQ(doc["status"].GetInt(), 2);
    ASSERT_EQ(doc["results"].Size(), 1u);
    EXPECT_EQ(doc["results"][0]["status"].GetInt(), 2);
    EXPECT_TRUE(doc["results"][0]["data"].IsNull());
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
    const std::string body = "{\"images\":[\"aGVsbG8=\"]}";
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

namespace {

double metrics_value(const std::string& body, const std::string& key) {
    // sample lines look like `name{model="..."} 42`; HELP/TYPE lines (`# HELP
    // name ...`) must not be parsed - match the key only when a label block
    // follows it
    size_t pos = 0;
    while ((pos = body.find(key, pos)) != std::string::npos) {
        const size_t after = pos + key.size();
        if (after < body.size() && body[after] == '{') {
            const auto value_pos = body.find(' ', after);
            if (value_pos != std::string::npos) {
                return std::atof(
                    body.substr(value_pos + 1, body.find('\n', value_pos)).c_str());
            }
        }
        pos = after;
    }
    ADD_FAILURE() << "metric sample line not found: " << key << "\n" << body;
    return -1.0;
}

}  // namespace

TEST(server_e2e_contract, queue_limit_returns_429_with_retry_after) {
    ServerHandle handle =
        start_server("model_run_timeout=5000\nfake_delay_ms=300\nmax_queue_depth=1\n");

    // request A occupies the single queue slot (300ms fake inference)
    HttpResp resp_a;
    std::thread sender_a([&handle, &resp_a]() {
        resp_a = send_request(handle.port, "POST", "/test/model",
                              "{\"images\":[\"aGVsbG8=\"],\"req_id\":\"a\"}",
                              k_json_auth_headers);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // request B arrives while the queue is full: fast-fail instead of queueing
    const auto resp_b =
        send_request(handle.port, "POST", "/test/model",
                     "{\"images\":[\"aGVsbG8=\"],\"req_id\":\"b\"}", k_json_auth_headers);
    sender_a.join();

    EXPECT_EQ(resp_a.status, 200);
    EXPECT_EQ(resp_b.status, 429);
    ASSERT_NE(resp_b.headers.find("retry-after"), resp_b.headers.end())
        << "overload responses must carry a Retry-After hint";
    const int retry_after = std::atoi(resp_b.headers.at("retry-after").c_str());
    EXPECT_GE(retry_after, 1);
    EXPECT_LE(retry_after, 60);

    const auto metrics = send_request(handle.port, "GET", "/metrics", "", {});
    EXPECT_GT(metrics_value(metrics.body, "mortred_queue_rejected_total"), 0.0);
}

TEST(server_e2e_contract, batch_collects_and_distributes_per_request_results) {
    ServerHandle handle = start_server(
        "model_run_timeout=5000\nfake_delay_ms=50\nmax_batch_size=4\nmax_batch_delay_ms=800\n");

    constexpr int kRequests = 4;
    std::vector<HttpResp> responses(kRequests);
    std::vector<std::thread> senders;
    for (int i = 0; i < kRequests; ++i) {
        // distinct payload sizes -> distinct echoed values: any cross-entry
        // mixup in the batch distribution would be visible in the response
        const std::string payload(static_cast<size_t>(10 + i * 7), 'a');
        const std::string body =
            "{\"images\":[\"" + payload + "\"],\"req_id\":\"batch-" + std::to_string(i) + "\"}";
        senders.emplace_back([&handle, &responses, i, body]() {
            responses[i] =
                send_request(handle.port, "POST", "/test/model", body, k_json_auth_headers);
        });
    }
    for (auto& sender : senders) {
        sender.join();
    }

    for (int i = 0; i < kRequests; ++i) {
        ASSERT_EQ(responses[i].status, 200) << "request " << i;
        const auto doc = parse_body(responses[i].body);
        ASSERT_FALSE(doc.HasParseError());
        EXPECT_EQ(doc["results"][0]["data"]["value"].GetInt(), 10 + i * 7)
            << "request " << i << " received another entry's result";
    }

    // all four requests went through the batch path: 4 observed items in
    // fewer than 4 batches proves real coalescing happened (exactly-one-
    // batch assertions would be timing-fragile on loaded CI machines)
    const auto metrics = send_request(handle.port, "GET", "/metrics", "", {});
    EXPECT_EQ(metrics_value(metrics.body, "mortred_batch_size_sum"), 4.0);
    const double batch_count = metrics_value(metrics.body, "mortred_batch_size_count");
    EXPECT_GE(batch_count, 1.0);
    EXPECT_LT(batch_count, 4.0) << "no coalescing happened: 4 batches of size 1";
}

TEST(server_e2e_contract, batch_timeout_returns_504) {
    // the batch waiter honors model_run_timeout exactly like the single path
    ServerHandle handle = start_server(
        "model_run_timeout=200\nfake_delay_ms=500\nmax_batch_size=4\nmax_batch_delay_ms=50\n");
    const auto resp = send_request(handle.port, "POST", "/test/model",
                                   "{\"images\":[\"aGVsbG8=\"],\"req_id\":\"slow\"}",
                                   k_json_auth_headers);
    EXPECT_EQ(resp.status, 504);
}

// ===== async job endpoints (P0-2) =====

TEST(server_e2e_contract, async_submit_returns_202_with_job_id) {
    ServerHandle handle = start_server("async_enabled=true\nfake_delay_ms=100\n");
    const auto resp =
        send_request(handle.port, "POST", "/jobs",
                     "{\"images\":[\"aGVsbG8=\"],\"req_id\":\"async-1\"}", k_json_auth_headers);
    EXPECT_EQ(resp.status, 202);
    auto doc = parse_body(resp.body);
    ASSERT_FALSE(doc.HasParseError());
    ASSERT_TRUE(doc.HasMember("job_id"));
    EXPECT_TRUE(doc["job_id"].IsString());
    EXPECT_NE(std::string(doc["job_id"].GetString()).find("job_"), std::string::npos);
    EXPECT_STREQ(doc["state"].GetString(), "pending");
}

TEST(server_e2e_contract, async_lifecycle_pending_to_done_to_result) {
    ServerHandle handle = start_server("async_enabled=true\nfake_delay_ms=200\n");
    const auto submit =
        send_request(handle.port, "POST", "/jobs",
                     "{\"images\":[\"aGVsbG8=\"],\"req_id\":\"async-2\"}", k_json_auth_headers);
    ASSERT_EQ(submit.status, 202);
    auto submit_doc = parse_body(submit.body);
    const std::string job_id = submit_doc["job_id"].GetString();

    // poll: eventually done
    bool done = false;
    for (int i = 0; i < 20; ++i) {
        const auto status = send_request(handle.port, "GET", "/jobs/" + job_id, "", k_json_auth_headers);
        auto doc = parse_body(status.body);
        if (std::string(doc["state"].GetString()) == "done") {
            done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_TRUE(done) << "job did not reach done state";

    // result: standard envelope with model output
    const auto result =
        send_request(handle.port, "GET", "/jobs/" + job_id + "/result", "", k_json_auth_headers);
    EXPECT_EQ(result.status, 200);
    auto result_doc = parse_body(result.body);
    ASSERT_FALSE(result_doc.HasParseError());
    EXPECT_EQ(result_doc["status"].GetInt(), 0);
    ASSERT_EQ(result_doc["results"].Size(), 1u);
    EXPECT_EQ(result_doc["results"][0]["data"]["value"].GetInt(), 8);  // payload echo
}

TEST(server_e2e_contract, async_incomplete_result_returns_409) {
    ServerHandle handle = start_server("async_enabled=true\nfake_delay_ms=5000\n");
    const auto submit =
        send_request(handle.port, "POST", "/jobs",
                     "{\"images\":[\"aGVsbG8=\"]}", k_json_auth_headers);
    ASSERT_EQ(submit.status, 202);
    auto doc = parse_body(submit.body);
    const std::string job_id = doc["job_id"].GetString();
    // job should still be running (5s delay), so result -> 409
    // (racing: if the job somehow finished, 200 is also acceptable)
    const auto result =
        send_request(handle.port, "GET", "/jobs/" + job_id + "/result", "", k_json_auth_headers);
    if (result.status == 409) {
        auto err_doc = parse_body(result.body);
        EXPECT_TRUE(err_doc.HasMember("error"));
    } else {
        EXPECT_EQ(result.status, 200) << "expected 409 (running) or 200 (fast finish)";
    }
}

TEST(server_e2e_contract, async_nonexistent_job_returns_404) {
    ServerHandle handle = start_server("async_enabled=true\n");
    const auto resp =
        send_request(handle.port, "GET", "/jobs/nonexistent", "", k_json_auth_headers);
    EXPECT_EQ(resp.status, 404);
}

TEST(server_e2e_contract, async_disabled_returns_404) {
    // async_enabled defaults to false: /jobs should be 404
    ServerHandle handle = start_server();
    const auto resp =
        send_request(handle.port, "POST", "/jobs",
                     "{\"images\":[\"aGVsbG8=\"]}", k_json_auth_headers);
    EXPECT_EQ(resp.status, 404);
}

TEST(server_e2e_contract, async_long_poll_wait_returns_done) {
    ServerHandle handle = start_server("async_enabled=true\nfake_delay_ms=300\n");
    const auto submit =
        send_request(handle.port, "POST", "/jobs",
                     "{\"images\":[\"aGVsbG8=\"]}", k_json_auth_headers);
    ASSERT_EQ(submit.status, 202);
    auto doc = parse_body(submit.body);
    const std::string job_id = doc["job_id"].GetString();

    // long-poll with 5s timeout: should return "done" after ~300ms
    const auto wait = send_request(handle.port, "GET", "/jobs/" + job_id + "/wait?timeout=5000",
                                   "", k_json_auth_headers);
    EXPECT_EQ(wait.status, 200);
    auto wait_doc = parse_body(wait.body);
    EXPECT_STREQ(wait_doc["state"].GetString(), "done");
}

TEST(server_e2e_contract, async_queue_full_returns_429) {
    // max_queue=1 with a 2s model: two CONCURRENT submissions race for one
    // admission slot - the CAS must admit exactly one (202) and reject the
    // other (429 + error body). Concurrent submission is required because the
    // 202 reply is only flushed after the job's go task completes (Workflow
    // server-series semantics, unchanged from the previous implementation),
    // so a serial second submit would always observe a free queue.
    ServerHandle handle =
        start_server("async_enabled=true\nfake_delay_ms=2000\nasync_max_queue=1\n");
    HttpResp r1, r2;
    std::thread t1([&]() {
        r1 = send_request(handle.port, "POST", "/jobs",
                          "{\"images\":[\"aGVsbG8=\"]}", k_json_auth_headers);
    });
    std::thread t2([&]() {
        r2 = send_request(handle.port, "POST", "/jobs",
                          "{\"images\":[\"aGVsbG8=\"]}", k_json_auth_headers);
    });
    t1.join();
    t2.join();
    const int accepted = (r1.status == 202) + (r2.status == 202);
    const int rejected = (r1.status == 429) + (r2.status == 429);
    EXPECT_EQ(accepted, 1) << "r1=" << r1.status << " body=" << r1.body
                           << " r2=" << r2.status << " body=" << r2.body;
    EXPECT_EQ(rejected, 1) << "r1=" << r1.status << " body=" << r1.body
                           << " r2=" << r2.status << " body=" << r2.body;
    const std::string rejected_body = r1.status == 429 ? r1.body : r2.body;
    auto doc = parse_body(rejected_body);
    ASSERT_FALSE(doc.HasParseError()) << "body=" << rejected_body;
    EXPECT_TRUE(doc.HasMember("error")) << "body=" << rejected_body;
}

TEST(server_e2e_contract, batch_item_failure_isolated) {
    // good + bad + good in one batch: the bad item reports its own error,
    // its batch mates must still return their correct results
    ServerHandle handle = start_server(
        "model_run_timeout=5000\nfake_delay_ms=20\nmax_batch_size=4\nmax_batch_delay_ms=500\n");

    HttpResp resp_a;
    HttpResp resp_c;
    const std::string payload_a(17, 'a');
    const std::string payload_c(31, 'c');
    std::thread sender_a([&handle, &resp_a, &payload_a]() {
        resp_a = send_request(handle.port, "POST", "/test/model",
                              "{\"images\":[\"" + payload_a + "\"],\"req_id\":\"a\"}",
                              k_json_auth_headers);
    });
    std::thread sender_c([&handle, &resp_c, &payload_c]() {
        resp_c = send_request(handle.port, "POST", "/test/model",
                              "{\"images\":[\"" + payload_c + "\"],\"req_id\":\"c\"}",
                              k_json_auth_headers);
    });
    // the "fail-me" item rides in the same collection window
    const auto resp_b =
        send_request(handle.port, "POST", "/test/model",
                     "{\"images\":[\"fail-me\"],\"req_id\":\"b\"}", k_json_auth_headers);
    sender_a.join();
    sender_c.join();

    EXPECT_EQ(resp_b.status, 500);
    auto doc_b = parse_body(resp_b.body);
    ASSERT_FALSE(doc_b.HasParseError());
    ASSERT_EQ(doc_b["results"].Size(), 1u);
    EXPECT_EQ(doc_b["results"][0]["status"].GetInt(), 2);
    EXPECT_TRUE(doc_b["results"][0]["data"].IsNull()) << "failed items keep data:null";

    ASSERT_EQ(resp_a.status, 200) << "the failing item must not fail its batch mates";
    auto doc_a = parse_body(resp_a.body);
    ASSERT_FALSE(doc_a.HasParseError());
    EXPECT_EQ(doc_a["results"][0]["data"]["value"].GetInt(), 17);

    ASSERT_EQ(resp_c.status, 200) << "the failing item must not fail its batch mates";
    auto doc_c = parse_body(resp_c.body);
    ASSERT_FALSE(doc_c.HasParseError());
    EXPECT_EQ(doc_c["results"][0]["data"]["value"].GetInt(), 31);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
