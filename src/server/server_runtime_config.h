/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: server_runtime_config.h
* Date: 26-9-9
************************************************/

// TOML [*_SERVER] section -> runtime value object. Schema validity lives in
// server_config_schema.h; this file maps legal keys to values, defaults,
// clamps and warnings. Parsing is a pure function: it MUST NOT start threads.

#ifndef MORTRED_SERVER_RUNTIME_CONFIG_H
#define MORTRED_SERVER_RUNTIME_CONFIG_H

#include <cstdlib>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "toml/toml.hpp"

#include "common/listen_policy.h"
#include "common/request_size_limit.h"
#include "common/status_code.h"
#include "server/server_config_schema.h"

namespace jinq {
namespace server {
using jinq::common::StatusCode;
using jinq::common::k_default_request_size_limit_mb;

struct ServerRuntimeConfig {
    int max_connections = 200;
    int peer_resp_timeout = 15 * 1000;
    int compute_threads = -1;
    int handler_threads = 50;
    size_t request_size_limit = k_default_request_size_limit_mb;
    int model_run_timeout = 500;
    std::string stuck_worker_action = "log";
    int stuck_worker_threshold_times = 3;
    std::string auth_token;
    int rate_limit_qps = 0;
    int max_queue_depth = 0;
    int max_batch_size = 1;
    int max_batch_delay_ms = 5;
    size_t max_request_items = 16;
    bool async_enabled = false;
    int async_timeout = 300000;
    int async_max_queue = 16;
    int async_job_ttl_ms = 300000;
    int async_max_completed = 100;
    int ewma_seed_ms = 500;
};

/***
 * Parse and validate the worker count.
 * Missing (value_or falls back to 0), zero or negative are config errors —
 * an empty worker queue would hang do_work's unbounded wait_dequeue forever, so
 * return -1 and let the caller refuse to start.
 */
inline int parse_worker_nums(const toml::table& server_section) {
    auto worker_nums = static_cast<int>(server_section["worker_nums"].value_or<int64_t>(0));
    if (const char* env = std::getenv("MORTRED_WORKER_NUMS"); env != nullptr && *env != '\0') {
        try {
            worker_nums = std::stoi(env);
        } catch (...) {
            LOG(ERROR) << "invalid MORTRED_WORKER_NUMS: " << env;
            return -1;
        }
    }
    if (worker_nums <= 0) {
        LOG(ERROR) << "invalid worker_nums: " << worker_nums
                   << " (missing, zero or negative), requests would hang forever";
        return -1;
    }
    return worker_nums;
}

inline StatusCode parse_server_security_config(const toml::table& server_section,
                                               ServerRuntimeConfig& cfg) {
    cfg.auth_token = server_section["auth_token"].value_or<std::string>(cfg.auth_token);
    if (const char* env = std::getenv("MORTRED_AUTH_TOKEN"); env != nullptr && *env != '\0') {
        cfg.auth_token = env;
    }
    cfg.rate_limit_qps =
        static_cast<int>(server_section["rate_limit_qps"].value_or<int64_t>(cfg.rate_limit_qps));

    auto listen_host = server_section["host"].value_or<std::string>("127.0.0.1");
    if (const char* env = std::getenv("MORTRED_LISTEN_HOST"); env != nullptr && *env != '\0') {
        listen_host = env;
    }
    if (!jinq::common::listen_host_permitted(listen_host)) {
        LOG(ERROR) << "refusing to serve on " << listen_host
                   << " (MORTRED_EXPOSE=" << jinq::common::mortred_expose_mode()
                   << "); bind 127.0.0.1 or set MORTRED_EXPOSE=docker|unsafe";
        return StatusCode::SERVER_INIT_FAILED;
    }
    if (!jinq::common::is_loopback_host(listen_host) && cfg.auth_token.empty()) {
        LOG(ERROR) << "refusing to serve on non-loopback host " << listen_host
                   << " without auth_token configured";
        return StatusCode::SERVER_INIT_FAILED;
    }
    return StatusCode::OK;
}

inline StatusCode parse_server_runtime_config(const toml::table& server_section,
                                              ServerRuntimeConfig& cfg) {
    // Missing optional keys must keep in-class defaults. Do not value_or(0):
    // that silently turns max_connections/peer_resp_timeout/handler_threads
    // into zero and compute_threads into "no pool" instead of auto (-1).
    const ServerRuntimeConfig defaults{};
    cfg = defaults;

    std::string schema_err;
    std::vector<std::string> schema_warnings;
    if (!validate_server_section(server_section, &schema_err, &schema_warnings)) {
        LOG(ERROR) << "invalid server config: " << schema_err;
        return StatusCode::SERVER_INIT_FAILED;
    }
    for (const auto& warning : schema_warnings) {
        LOG(WARNING) << warning;
    }

    cfg.max_connections = static_cast<int>(
        server_section["max_connections"].value_or<int64_t>(defaults.max_connections));
    cfg.peer_resp_timeout =
        static_cast<int>(server_section["peer_resp_timeout"].value_or<int64_t>(
            defaults.peer_resp_timeout / 1000)) *
        1000;
    cfg.compute_threads = static_cast<int>(
        server_section["compute_threads"].value_or<int64_t>(defaults.compute_threads));
    cfg.handler_threads = static_cast<int>(
        server_section["handler_threads"].value_or<int64_t>(defaults.handler_threads));
    if (auto limit = server_section["request_size_limit"].value_or<int64_t>(0); limit > 0) {
        cfg.request_size_limit = static_cast<size_t>(limit);
    }
    cfg.model_run_timeout = static_cast<int>(
        server_section["model_run_timeout"].value_or<int64_t>(defaults.model_run_timeout));
    if (cfg.model_run_timeout <= 0) {
        LOG(WARNING) << "model_run_timeout <= 0: per-request timeout disabled; a hung model "
                     << "keeps its worker forever, subsequent requests block indefinitely and "
                     << "clients may never receive a response";
    }
    cfg.stuck_worker_action =
        server_section["stuck_worker_action"].value_or<std::string>(defaults.stuck_worker_action);
    cfg.stuck_worker_threshold_times = static_cast<int>(
        server_section["stuck_worker_threshold_times"].value_or<int64_t>(
            defaults.stuck_worker_threshold_times));
    if (cfg.stuck_worker_threshold_times <= 0) {
        cfg.stuck_worker_threshold_times = defaults.stuck_worker_threshold_times;
    }
    cfg.max_queue_depth = static_cast<int>(
        server_section["max_queue_depth"].value_or<int64_t>(defaults.max_queue_depth));
    if (cfg.max_queue_depth < 0) {
        LOG(WARNING) << "max_queue_depth < 0: queue depth limit disabled";
        cfg.max_queue_depth = defaults.max_queue_depth;
    }
    cfg.max_batch_size = static_cast<int>(
        server_section["max_batch_size"].value_or<int64_t>(defaults.max_batch_size));
    if (cfg.max_batch_size < 1) {
        LOG(WARNING) << "max_batch_size < 1: batching disabled";
        cfg.max_batch_size = defaults.max_batch_size;
    }
    cfg.max_batch_delay_ms = static_cast<int>(
        server_section["max_batch_delay_ms"].value_or<int64_t>(defaults.max_batch_delay_ms));
    if (cfg.max_batch_delay_ms < 0) {
        LOG(WARNING) << "max_batch_delay_ms < 0: using the 5ms default";
        cfg.max_batch_delay_ms = defaults.max_batch_delay_ms;
    }
    cfg.max_request_items = static_cast<size_t>(
        server_section["max_request_items"].value_or<int64_t>(
            static_cast<int64_t>(defaults.max_request_items)));
    if (cfg.max_request_items < 1) {
        LOG(WARNING) << "max_request_items < 1: using the 16 default";
        cfg.max_request_items = defaults.max_request_items;
    }
    cfg.ewma_seed_ms = cfg.model_run_timeout > 0 ? cfg.model_run_timeout : defaults.ewma_seed_ms;
    cfg.async_enabled = server_section["async_enabled"].value_or<bool>(defaults.async_enabled);
    cfg.async_timeout = static_cast<int>(
        server_section["async_timeout"].value_or<int64_t>(defaults.async_timeout));
    cfg.async_max_queue = static_cast<int>(
        server_section["async_max_queue"].value_or<int64_t>(defaults.async_max_queue));
    cfg.async_job_ttl_ms = static_cast<int>(
        server_section["async_job_ttl"].value_or<int64_t>(defaults.async_job_ttl_ms));
    cfg.async_max_completed = static_cast<int>(
        server_section["async_max_completed"].value_or<int64_t>(defaults.async_max_completed));
    if (cfg.async_enabled) {
        LOG(INFO) << "async jobs enabled: timeout=" << cfg.async_timeout
                  << "ms, max_queue=" << cfg.async_max_queue
                  << ", job_ttl=" << cfg.async_job_ttl_ms << "ms";
    }
    if (cfg.max_batch_size > 1) {
        LOG(INFO) << "dynamic batching enabled: max_batch_size=" << cfg.max_batch_size
                  << ", max_batch_delay_ms=" << cfg.max_batch_delay_ms;
    }
    return parse_server_security_config(server_section, cfg);
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_RUNTIME_CONFIG_H
