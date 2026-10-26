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

#include "apps/common/exit_codes.h"
#include "common/file_path_util.h"
#include "apps/common/listen_port.h"
#include "toml/toml.hpp"

#include "common/process_stop.h"
#include "server/abstract_server.h"

namespace mortred {
namespace apps {
using mortred::common::StatusCode;

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
    const std::function<std::unique_ptr<mortred::server::BaseAiServer>(const std::string&)>&
        make_server) {
    // Arm before glog/Workflow threads so they inherit the blocked SIGINT/SIGTERM
    // mask. Usage-only invocations skip arm() and never wait().
    mortred::common::ProcessStop process_stop;
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
        return kExitUsage;
    }

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config_parsed = toml::parse_file(config_file_path);
    if (!config_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(config_parsed.error().description());
        return kExitConfig;
    }
    auto config = std::move(config_parsed).table();
    // A configured weight path that does not exist is a config error, not a
    // model-init failure: report it here, before any session is built, naming
    // the table it came from. Relative paths resolve against the working
    // directory, exactly as the backends resolve them.
    for (const auto &[table_key, table_node] : config) {
        if (!table_node.is_table()) {
            continue;
        }
        const auto *backend = table_node.as_table()->get("backend");
        if (backend == nullptr || !backend->is_table()) {
            continue;
        }
        const auto *path_node = backend->as_table()->get("model_file_path");
        if (path_node == nullptr) {
            continue;  // the backend parser reports the missing key with a status
        }
        const std::string weight_path = path_node->value_or<std::string>("");
        if (weight_path.empty()) {
            continue;
        }
        if (!mortred::common::FilePathUtil::is_file_exist(weight_path)) {
            LOG(ERROR) << "model_file_path does not exist: " << weight_path << " (["
                       << table_key.str() << ".backend])";
            return kExitConfig;
        }
    }
    std::string section_alias;
    const auto* server_cfg_ptr = find_section_ci(config, server_section, &section_alias);
    if (server_cfg_ptr == nullptr) {
        LOG(ERROR) << "server section [" << server_section << "] missing from " << config_file_path;
        return kExitConfig;
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
            port = std::stoll(env);
        } catch (...) {
            LOG(ERROR) << "MORTRED_LISTEN_PORT is not an integer: " << env;
            return kExitConfig;
        }
    }
    // the port is narrowed to unsigned short for listen(), so reject what would
    // silently wrap into a different port instead of failing at bind time
    if (!valid_listen_port(port)) {
        LOG(ERROR) << "invalid listen port " << port
                   << " (expected 1..65535; set [" << server_section
                   << "] port or MORTRED_LISTEN_PORT)";
        return kExitConfig;
    }
    LOG(INFO) << "serve on port: " << port;

    // factory registration key: process-internal registry, no external consumer
    auto server = make_server("server");
    auto status = server->init(config);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "server init failed, status: "
                   << std::to_string(static_cast<int>(status));
        return kExitSoftware;
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
    return kExitSoftware;
}

}  // namespace apps
}  // namespace mortred

#endif  // MORTRED_APPS_MODEL_SERVER_MAIN_H
