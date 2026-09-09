/************************************************
 * Author: Codex
 * File: server_runtime_config_unittest.cc
 * Date: 2026-09-09
 ************************************************/

#include <gtest/gtest.h>
#include <toml/toml.hpp>

#include "common/status_code.h"
#include "server/server_runtime_config.h"

using jinq::common::StatusCode;
using jinq::server::ServerRuntimeConfig;
using jinq::server::parse_server_runtime_config;

namespace {

toml::table parse_section(const char* text) {
    return std::move(toml::parse(text)).table();
}

}  // namespace

TEST(server_runtime_config, missing_optional_keys_keep_struct_defaults) {
    ServerRuntimeConfig cfg;
    cfg.max_connections = 1;
    cfg.peer_resp_timeout = 1;
    cfg.compute_threads = 7;
    cfg.handler_threads = 1;
    cfg.request_size_limit = 1;
    cfg.model_run_timeout = 1;
    cfg.stuck_worker_action = "exit";
    cfg.stuck_worker_threshold_times = 9;
    cfg.rate_limit_qps = 9;
    cfg.max_queue_depth = 9;
    cfg.max_batch_size = 8;
    cfg.max_batch_delay_ms = 99;
    cfg.max_request_items = 2;
    cfg.async_enabled = true;
    cfg.async_timeout = 1;
    cfg.async_max_queue = 1;
    cfg.async_job_ttl_ms = 1;
    cfg.async_max_completed = 1;
    cfg.ewma_seed_ms = 1;

    auto section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 1
)");
    ASSERT_EQ(parse_server_runtime_config(section, cfg), StatusCode::OK);

    const ServerRuntimeConfig expected;
    EXPECT_EQ(cfg.max_connections, expected.max_connections);
    EXPECT_EQ(cfg.peer_resp_timeout, expected.peer_resp_timeout);
    EXPECT_EQ(cfg.compute_threads, expected.compute_threads);
    EXPECT_EQ(cfg.handler_threads, expected.handler_threads);
    EXPECT_EQ(cfg.request_size_limit, expected.request_size_limit);
    EXPECT_EQ(cfg.model_run_timeout, expected.model_run_timeout);
    EXPECT_EQ(cfg.stuck_worker_action, expected.stuck_worker_action);
    EXPECT_EQ(cfg.stuck_worker_threshold_times, expected.stuck_worker_threshold_times);
    EXPECT_TRUE(cfg.auth_token.empty());
    EXPECT_EQ(cfg.rate_limit_qps, expected.rate_limit_qps);
    EXPECT_EQ(cfg.max_queue_depth, expected.max_queue_depth);
    EXPECT_EQ(cfg.max_batch_size, expected.max_batch_size);
    EXPECT_EQ(cfg.max_batch_delay_ms, expected.max_batch_delay_ms);
    EXPECT_EQ(cfg.max_request_items, expected.max_request_items);
    EXPECT_EQ(cfg.async_enabled, expected.async_enabled);
    EXPECT_EQ(cfg.async_timeout, expected.async_timeout);
    EXPECT_EQ(cfg.async_max_queue, expected.async_max_queue);
    EXPECT_EQ(cfg.async_job_ttl_ms, expected.async_job_ttl_ms);
    EXPECT_EQ(cfg.async_max_completed, expected.async_max_completed);
    EXPECT_EQ(cfg.ewma_seed_ms, expected.ewma_seed_ms);
}

TEST(server_runtime_config, missing_required_key_fails) {
    ServerRuntimeConfig cfg;
    auto section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
)");
    EXPECT_EQ(parse_server_runtime_config(section, cfg), StatusCode::SERVER_INIT_FAILED);
}

TEST(server_runtime_config, defaults_and_clamps) {
    ServerRuntimeConfig cfg;
    auto section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 2
model_run_timeout = 800
max_batch_size = 0
max_batch_delay_ms = -3
max_request_items = 0
stuck_worker_threshold_times = 0
max_queue_depth = -8
)");
    ASSERT_EQ(parse_server_runtime_config(section, cfg), StatusCode::OK);
    EXPECT_EQ(cfg.max_connections, 200);
    EXPECT_EQ(cfg.peer_resp_timeout, 15 * 1000);
    EXPECT_EQ(cfg.compute_threads, -1);
    EXPECT_EQ(cfg.handler_threads, 50);
    EXPECT_EQ(cfg.model_run_timeout, 800);
    EXPECT_EQ(cfg.ewma_seed_ms, 800);
    EXPECT_EQ(cfg.max_batch_size, 1);
    EXPECT_EQ(cfg.max_batch_delay_ms, 5);
    EXPECT_EQ(cfg.max_request_items, 16);
    EXPECT_EQ(cfg.stuck_worker_threshold_times, 3);
    EXPECT_EQ(cfg.max_queue_depth, 0);
    EXPECT_EQ(cfg.stuck_worker_action, "log");
    EXPECT_FALSE(cfg.async_enabled);
    EXPECT_EQ(cfg.async_timeout, 300000);
}

TEST(server_runtime_config, async_and_batch_flags) {
    ServerRuntimeConfig cfg;
    auto section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 1
async_enabled = true
async_timeout = 12000
async_max_queue = 4
async_job_ttl = 9000
async_max_completed = 7
max_batch_size = 4
max_batch_delay_ms = 11
stuck_worker_action = "exit"
)");
    ASSERT_EQ(parse_server_runtime_config(section, cfg), StatusCode::OK);
    EXPECT_TRUE(cfg.async_enabled);
    EXPECT_EQ(cfg.async_timeout, 12000);
    EXPECT_EQ(cfg.async_max_queue, 4);
    EXPECT_EQ(cfg.async_job_ttl_ms, 9000);
    EXPECT_EQ(cfg.async_max_completed, 7);
    EXPECT_EQ(cfg.max_batch_size, 4);
    EXPECT_EQ(cfg.max_batch_delay_ms, 11);
    EXPECT_EQ(cfg.stuck_worker_action, "exit");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
