/************************************************
 * Author: Codex
 * File: server_runtime_config_unittest.cc
 * Date: 2026-09-09
 ************************************************/

#include <gtest/gtest.h>
#include <toml/toml.hpp>

#include "common/status_code.h"
#include "server/server_runtime_config.h"

using mortred::common::StatusCode;
using mortred::server::ServerRuntimeConfig;
using mortred::server::k_model_run_timeout_clamp_ms;
using mortred::server::parse_server_runtime_config;

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
    EXPECT_EQ(cfg.receive_timeout, expected.receive_timeout);
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
    EXPECT_EQ(cfg.receive_timeout, 60 * 1000);
    EXPECT_EQ(cfg.compute_threads, -1);
    // Round 13 lowered the shipped handler-thread default from 50 to 8: 50
    // handler threads per model process dominated the idle footprint (the
    // capacity work). A config that omits the key must get the new default.
    EXPECT_EQ(cfg.handler_threads, 8);
    EXPECT_EQ(cfg.model_run_timeout, 800);
    EXPECT_EQ(cfg.ewma_seed_ms, 800);
    EXPECT_EQ(cfg.max_batch_size, 1);
    EXPECT_EQ(cfg.max_batch_delay_ms, 5);
    EXPECT_EQ(cfg.max_request_items, 16);
    EXPECT_EQ(cfg.stuck_worker_threshold_times, 3);
    // Round 14: a non-positive queue depth is clamped to the positive default
    // instead of silently disabling the admission bound (max_queue_depth = -8
    // in this fixture).
    EXPECT_EQ(cfg.max_queue_depth, 32);
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


TEST(server_runtime_config, receive_timeout_seconds_to_ms_and_reject_unbounded) {
    ServerRuntimeConfig cfg;
    auto section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 1
receive_timeout = 30
)");
    ASSERT_EQ(parse_server_runtime_config(section, cfg), StatusCode::OK);
    EXPECT_EQ(cfg.receive_timeout, 30 * 1000);

    ServerRuntimeConfig cfg2;
    auto section2 = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 1
receive_timeout = 0
)");
    ASSERT_EQ(parse_server_runtime_config(section2, cfg2), StatusCode::OK);
    EXPECT_EQ(cfg2.receive_timeout, 60 * 1000);
}

TEST(server_runtime_config, model_run_timeout_non_positive_clamps_to_the_bounded_value) {
    // Three states of the same key: an explicit 0 and an explicit -1 are both
    // clamped to the named bound, because every request must own a deadline and
    // a request timer; an absent key keeps the 500ms struct default.
    // The bound is deliberately NOT that default: the ids that configured 0
    // (SAM AMG, diffusion) asked for a long budget, and 500ms would turn them
    // into 504 machines.
    EXPECT_EQ(k_model_run_timeout_clamp_ms, 300000);

    ServerRuntimeConfig zero;
    auto zero_section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 1
model_run_timeout = 0
)");
    ASSERT_EQ(parse_server_runtime_config(zero_section, zero), StatusCode::OK);
    EXPECT_EQ(zero.model_run_timeout, 300000);
    // the applied timeout also seeds the run-time EWMA (pre-existing coupling)
    EXPECT_EQ(zero.ewma_seed_ms, 300000);

    ServerRuntimeConfig negative;
    auto negative_section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 1
model_run_timeout = -1
)");
    ASSERT_EQ(parse_server_runtime_config(negative_section, negative), StatusCode::OK);
    EXPECT_EQ(negative.model_run_timeout, 300000);

    ServerRuntimeConfig absent;
    auto absent_section = parse_section(R"(
host = "127.0.0.1"
server_uri = "/test"
port = 8080
worker_nums = 1
)");
    ASSERT_EQ(parse_server_runtime_config(absent_section, absent), StatusCode::OK);
    EXPECT_EQ(absent.model_run_timeout, 500);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
