/************************************************
 * Author: Codex
 * File: catalog.cpp
 *
 * Web console server registry. The registry is derived from the TOML configs
 * under conf/server only: each [*_SERVER] section must declare `server_exe` (plus port / host /
 * server_uri), so the config -> executable mapping is explicit and can never
 * silently go stale. The previous token-overlap heuristic and the hard-coded
 * add_missing_server entries were removed (see scripts/check_consistency.py
 * check_server_exe_mapping for the bidirectional coverage gate).
 ************************************************/

#include "catalog.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace mortred_web {

namespace {

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

/***
 * minimal TOML parser: [section] + key=value lines (comments start with #)
 */
std::map<std::string, std::map<std::string, std::string>> parse_toml(const std::string& content) {
    std::map<std::string, std::map<std::string, std::string>> cfg;
    std::string section;
    std::istringstream ss(content);
    std::string line;
    while (std::getline(ss, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }
        if (t.front() == '[' && t.back() == ']') {
            section = t.substr(1, t.size() - 2);
            continue;
        }
        auto eq = t.find('=');
        if (eq == std::string::npos || section.empty()) {
            continue;
        }
        std::string key = trim(t.substr(0, eq));
        std::string val = unquote(trim(t.substr(eq + 1)));
        cfg[section][key] = val;
    }
    return cfg;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
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

bool Catalog::init(const std::string& project_root) {
    _entries.clear();
    std::string conf_dir = project_root + "/conf/server";

    std::error_code ec;
    if (!fs::exists(conf_dir, ec)) {
        return false;
    }

    for (const auto& entry : fs::recursive_directory_iterator(conf_dir, ec)) {
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".toml") {
            continue;
        }
        std::string cfg_path = entry.path().string();
        std::string content = read_file(cfg_path);
        if (content.find("server_uri") == std::string::npos) {
            continue;
        }

        auto cfg = parse_toml(content);
        std::string section;
        std::string host = "localhost";
        std::string auth_token;
        std::string exe;
        std::string uri;
        int port = 0;
        for (const auto& [sec, kv] : cfg) {
            auto pit = kv.find("port");
            if (pit == kv.end()) {
                continue;
            }
            section = sec;
            try {
                port = std::stoi(pit->second);
            } catch (...) {
                continue;
            }
            auto hit = kv.find("host");
            if (hit != kv.end()) {
                host = hit->second;
            }
            auto ait = kv.find("auth_token");
            if (ait != kv.end()) {
                auth_token = ait->second;
            }
            auto uit = kv.find("server_uri");
            if (uit == kv.end()) {
                continue;
            }
            uri = uit->second;
            auto eit = kv.find("server_exe");
            if (eit == kv.end() || eit->second.empty()) {
                std::fprintf(stderr,
                             "[catalog] %s: server section [%s] has no server_exe, skipped "
                             "(add server_exe=\"<exe>.out\" to the config)\n",
                             cfg_path.c_str(), sec.c_str());
                section.clear();
            } else {
                exe = eit->second;
            }
            break;
        }
        if (section.empty() || port <= 0 || uri.empty() || exe.empty() || exe.size() <= 4) {
            continue;
        }

        std::string id = exe.substr(0, exe.size() - 4);
        bool dup = false;
        for (const auto& e : _entries) {
            if (e.id == id) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        std::string type = to_lower(section).find("chat") != std::string::npos ? "chat" : "image";
        std::string category = "other";
        fs::path rel = fs::relative(cfg_path, conf_dir, ec);
        if (!rel.empty()) {
            auto it = rel.begin();
            if (it != rel.end()) {
                category = *it;
            }
        }

        ServerEntry e;
        e.id = id;
        e.name = id;
        e.category = category;
        e.exe = exe;
        e.config = cfg_path;
        e.host = host;
        e.auth_token = auth_token;
        e.port = port;
        e.uri = uri;
        e.type = type;
        _entries.push_back(std::move(e));
    }
    return true;
}

}  // namespace mortred_web
