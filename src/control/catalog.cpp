/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: catalog.cpp
* Date: 26-8-22
************************************************/

#include "control/catalog.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>

#include "control/mini_toml.h"

namespace fs = std::filesystem;

namespace mortred {
namespace control {

namespace {

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

}  // namespace

const ServerEntry* Catalog::find(const std::string& id) const {
    for (const auto& e : _entries) {
        if (e.id == id) {
            return &e;
        }
    }
    return nullptr;
}

const ServerEntry* Catalog::find_by_uri(const std::string& uri) const {
    for (const auto& e : _entries) {
        if (e.uri == uri) {
            return &e;
        }
    }
    return nullptr;
}

bool Catalog::init(const std::string& project_root, std::string* err,
                   const std::string& profile) {
    _entries.clear();
    const std::string runtime_profile =
        (profile == "cpu") ? "cpu" : "gpu";  // unknown values fall back to gpu
    const std::string conf_dir = project_root + "/conf/server";

    std::error_code ec;
    if (!fs::exists(conf_dir, ec)) {
        if (err != nullptr) {
            *err = "conf/server dir not found under " + project_root;
        }
        return false;
    }

    std::set<std::string> seen_ids;
    std::set<int> seen_ports;
    std::set<std::string> seen_uris;

    for (const auto& file : fs::recursive_directory_iterator(conf_dir, ec)) {
        if (!file.is_regular_file(ec) || file.path().extension() != ".toml") {
            continue;
        }
        const std::string cfg_path = file.path().string();
        mini_toml::Doc doc;
        if (!mini_toml::load(cfg_path, &doc)) {
            continue;
        }

        // exactly one [*_SERVER] section per server config (consistency-checked)
        const auto* kv = static_cast<const mini_toml::Table*>(nullptr);
        std::string section;
        for (const auto& [sec, table] : doc) {
            if (sec.size() > 7 && sec.compare(sec.size() - 7, 7, "_SERVER") == 0) {
                kv = &table;
                section = sec;
                break;
            }
        }
        if (kv == nullptr) {
            continue;
        }

        ServerEntry e;
        e.config = cfg_path;
        // profile filter happens BEFORE any duplicate checks: cpu/gpu variants
        // of the same model (same exe/port) may coexist as separate files but
        // only one variant set is active per catalog run
        e.profile = kv->count("profile") != 0 ? kv->at("profile") : "gpu";
        if (e.profile != "any" && e.profile != runtime_profile) {
            continue;
        }
        e.host = kv->count("host") != 0 ? kv->at("host") : "localhost";
        e.port = kv->count("port") != 0 ? mini_toml::to_int(kv->at("port"), 0) : 0;
        e.uri = kv->count("server_uri") != 0 ? kv->at("server_uri") : "";
        e.exe = kv->count("server_exe") != 0 ? kv->at("server_exe") : "";

        if (e.exe.empty()) {
            std::fprintf(stderr, "[catalog] %s: [%s] has no server_exe, skipped\n",
                         cfg_path.c_str(), section.c_str());
            continue;
        }
        if (e.exe.size() <= 4 || e.exe.compare(e.exe.size() - 4, 4, ".out") != 0) {
            std::fprintf(stderr, "[catalog] %s: server_exe must end with .out: '%s'\n",
                         cfg_path.c_str(), e.exe.c_str());
            continue;
        }
        if (e.port <= 0 || e.uri.empty() || e.uri[0] != '/') {
            if (err != nullptr) {
                *err = cfg_path + ": invalid port/server_uri in [" + section + "]";
            }
            return false;
        }

        e.id = e.exe.substr(0, e.exe.size() - 4);
        if (seen_ids.count(e.id) != 0) {
            if (err != nullptr) {
                *err = "duplicate server id: " + e.id;
            }
            return false;
        }
        if (seen_ports.count(e.port) != 0) {
            if (err != nullptr) {
                *err = "duplicate model server port: " + std::to_string(e.port);
            }
            return false;
        }
        if (seen_uris.count(e.uri) != 0) {
            if (err != nullptr) {
                *err = "duplicate server_uri (gateway routing key): " + e.uri;
            }
            return false;
        }
        seen_ids.insert(e.id);
        seen_ports.insert(e.port);
        seen_uris.insert(e.uri);

        e.name = e.id;
        e.type = to_lower(section).find("chat") != std::string::npos ? "chat" : "image";
        e.category = "other";
        const fs::path rel = fs::relative(cfg_path, conf_dir, ec);
        if (!rel.empty() && rel.begin() != rel.end()) {
            e.category = rel.begin()->string();
        }
        _entries.push_back(std::move(e));
    }
    return true;
}

}  // namespace control
}  // namespace mortred
