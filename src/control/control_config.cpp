/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: control_config.cpp
* Date: 26-8-22
************************************************/

#include "control/control_config.h"

#include "control/mini_toml.h"
#include "control/restart_policy.h"

namespace mortred {
namespace control {

ServerPolicy ControlConfig::effective_policy(const std::string& id) const {
    ServerPolicy p;
    const auto it = servers.find(id);
    if (it != servers.end()) {
        p = it->second;
    }
    if (!p.has_autostart) {
        p.autostart = supervisor.autostart_default;
    }
    if (!p.has_restart_policy) {
        p.restart_policy = "on-failure";
    }
    return p;
}

namespace {

bool read_int(const mini_toml::Table& kv, const std::string& key, int* out, int min, int max,
              const std::string& ctx, std::string* err) {
    if (kv.count(key) == 0) {
        return true;
    }
    const int v = mini_toml::to_int(kv.at(key), min - 1);
    if (v < min || v > max) {
        if (err != nullptr) {
            *err = ctx + ": '" + key + "' must be an integer in [" + std::to_string(min) + ", " +
                   std::to_string(max) + "]";
        }
        return false;
    }
    *out = v;
    return true;
}

bool read_str(const mini_toml::Table& kv, const std::string& key, std::string* out,
              const std::string& ctx, std::string* err) {
    if (kv.count(key) == 0) {
        return true;
    }
    const std::string v = kv.at(key);
    if (v.empty()) {
        if (err != nullptr) {
            *err = ctx + ": '" + key + "' must be a non-empty string";
        }
        return false;
    }
    *out = v;
    return true;
}

}  // namespace

bool ControlConfig::load(const std::string& path, ControlConfig* out, std::string* err) {
    mini_toml::Doc doc;
    if (!mini_toml::load(path, &doc)) {
        if (err != nullptr) {
            *err = "cannot open control config: " + path;
        }
        return false;
    }

    ControlConfig cfg;

    if (doc.count("supervisor") != 0) {
        const auto& kv = doc.at("supervisor");
        const std::string ctx = "[supervisor]";
        if (!read_str(kv, "api_host", &cfg.supervisor.api_host, ctx, err) ||
            !read_str(kv, "log_dir", &cfg.supervisor.log_dir, ctx, err) ||
            !read_str(kv, "bin_dir", &cfg.supervisor.bin_dir, ctx, err) ||
            !read_str(kv, "lib_dir", &cfg.supervisor.lib_dir, ctx, err) ||
            !read_str(kv, "libs_dir", &cfg.supervisor.libs_dir, ctx, err) ||
            !read_int(kv, "api_port", &cfg.supervisor.api_port, 1, 65535, ctx, err) ||
            !read_int(kv, "start_concurrency", &cfg.supervisor.start_concurrency, 1, 32, ctx, err) ||
            !read_int(kv, "log_rotate_mb", &cfg.supervisor.log_rotate_mb, 1, 4096, ctx, err)) {
            return false;
        }
        if (kv.count("autostart_default") != 0) {
            cfg.supervisor.autostart_default =
                mini_toml::to_bool(kv.at("autostart_default"), false);
        }
    }

    if (doc.count("gateway") != 0) {
        const auto& kv = doc.at("gateway");
        const std::string ctx = "[gateway]";
        if (!read_str(kv, "host", &cfg.gateway.host, ctx, err) ||
            !read_int(kv, "port", &cfg.gateway.port, 1, 65535, ctx, err) ||
            !read_int(kv, "request_size_limit_mb", &cfg.gateway.request_size_limit_mb, 1, 4096,
                      ctx, err) ||
            !read_int(kv, "max_connections", &cfg.gateway.max_connections, 1, 100000, ctx, err) ||
            !read_int(kv, "upstream_send_timeout_ms", &cfg.gateway.upstream_send_timeout_ms, 1000,
                      3600000, ctx, err) ||
            !read_int(kv, "upstream_recv_timeout_ms", &cfg.gateway.upstream_recv_timeout_ms, 1000,
                      3600000, ctx, err)) {
            return false;
        }
    }

    for (const auto& [section, kv] : doc) {
        if (section.compare(0, 8, "servers.") != 0) {
            continue;
        }
        const std::string id = section.substr(8);
        if (id.empty()) {
            if (err != nullptr) {
                *err = "[" + section + "]: empty server id";
            }
            return false;
        }
        ServerPolicy p;
        if (kv.count("enabled") != 0) {
            p.enabled = mini_toml::to_bool(kv.at("enabled"), true);
        }
        if (kv.count("autostart") != 0) {
            p.has_autostart = true;
            p.autostart = mini_toml::to_bool(kv.at("autostart"), false);
        }
        if (kv.count("restart_policy") != 0) {
            const std::string v = kv.at("restart_policy");
            RestartPolicyKind kind = RestartPolicyKind::kOnFailure;
            if (!parse_restart_policy(v, &kind)) {
                if (err != nullptr) {
                    *err = "[" + section + "]: restart_policy must be on-failure|always|no, got '" +
                           v + "'";
                }
                return false;
            }
            p.has_restart_policy = true;
            p.restart_policy = v;
        }
        cfg.servers[id] = p;
    }

    *out = cfg;
    return true;
}

}  // namespace control
}  // namespace mortred
