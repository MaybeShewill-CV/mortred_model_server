/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: model_server_main.h
* Date: 26-8-19
************************************************/

// Shared main() body for all model server executables: glog setup, config
// parsing, host/port extraction, ProcessStop (SIGINT/SIGTERM), start and wait.
#ifndef MORTRED_APPS_MODEL_SERVER_MAIN_H
#define MORTRED_APPS_MODEL_SERVER_MAIN_H

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

#include <glog/logging.h>
#include "toml/toml.hpp"

#include "common/process_stop.h"
#include "server/abstract_server.h"

namespace jinq {
namespace apps {
using jinq::common::StatusCode;

/*** Table lookup that tolerates a case difference between the catalog id and
 * the TOML table name (legacy conf/model tables are uppercase, product ids
 * lowercase). Exact match wins; the first case-insensitive match is returned
 * and *matched_name reports the actual key so callers can log the alias. */
inline const toml::table* find_section_ci(const toml::table& config, const std::string& name,
                                          std::string* matched_name = nullptr) {
    if (auto* node = config.get(name); node != nullptr && node->is_table()) {
        if (matched_name != nullptr) {
            *matched_name = name;
        }
        return node->as_table();
    }
    std::string wanted = name;
    std::transform(wanted.begin(), wanted.end(), wanted.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    for (auto&& [key, value] : config) {
        if (!value.is_table()) {
            continue;
        }
        std::string candidate(key.str());
        std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (candidate == wanted) {
            if (matched_name != nullptr) {
                *matched_name = std::string(key.str());
            }
            return value.as_table();
        }
    }
    return nullptr;
}

inline int run_model_server_main(
    int argc, char** argv,
    const std::string& server_section,
    const std::function<std::unique_ptr<jinq::server::BaseAiServer>(const std::string&)>&
        make_server) {
    // Arm before glog/Workflow threads so they inherit the blocked SIGINT/SIGTERM
    // mask. Usage-only invocations skip arm() and never wait().
    jinq::common::ProcessStop process_stop;
    if (argc == 2) {
        process_stop.arm();
    }

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    if (argc != 2) {
        LOG(INFO) << "usage:";
        LOG(INFO) << "exe cfg_path";
        return -1;
    }

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config_parsed = toml::parse_file(config_file_path);
    if (!config_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(config_parsed.error().description());
        return -1;
    }
    auto config = std::move(config_parsed).table();
    std::string section_alias;
    const auto* server_cfg_ptr = find_section_ci(config, server_section, &section_alias);
    if (server_cfg_ptr == nullptr) {
        LOG(ERROR) << "server section [" << server_section << "] missing from " << config_file_path;
        return -1;
    }
    if (section_alias != server_section) {
        LOG(WARNING) << "config table [" << section_alias << "] matches server section ["
                     << server_section << "] case-insensitively";
    }
    const auto& server_cfg = *server_cfg_ptr;
    auto port = server_cfg["port"].value_or<int64_t>(0);
    auto host = server_cfg["host"].value_or<std::string>("127.0.0.1");
    // environment overrides (highest precedence): the supervisor injects these
    // to force loopback binding + internal auth for managed children
    if (const char* env = std::getenv("MORTRED_LISTEN_HOST"); env != nullptr && *env != '\0') {
        host = env;
    }
    if (const char* env = std::getenv("MORTRED_LISTEN_PORT"); env != nullptr && *env != '\0') {
        try {
            const int env_port = std::stoi(env);
            if (env_port > 0 && env_port <= 65535) {
                port = env_port;
            }
        } catch (...) {
            LOG(WARNING) << "ignoring invalid MORTRED_LISTEN_PORT: " << env;
        }
    }
    LOG(INFO) << "serve on port: " << port;

    // factory registration key: process-internal registry, no external consumer
    auto server = make_server("server");
    auto status = server->init(config);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "server init failed, status: "
                   << std::to_string(static_cast<int>(status));
        return -1;
    }
    const int rc = server->start(host.c_str(), static_cast<unsigned short>(port));
    if (rc == 0) {
        process_stop.wait();
        // drain before stop: /ready -> 503 so the gateway/LB pulls this
        // instance while in-flight requests finish, then cut the socket
        server->begin_drain();
        server->stop();
        return 0;
    }
    LOG(ERROR) << "Cannot start server on " << host << ":" << port;
    if (errno != 0) {
        LOG(ERROR) << "listen errno=" << errno << ": " << std::strerror(errno);
    } else {
        LOG(ERROR) << "listen failed (port already in use is the usual cause; "
                   << "stop mortred-supervisor or another process on this port)";
    }
    return -1;
}

}  // namespace apps
}  // namespace jinq

#endif  // MORTRED_APPS_MODEL_SERVER_MAIN_H
