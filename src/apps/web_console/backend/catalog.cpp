/************************************************
 * Author: Codex
 * File: catalog.cpp
 ************************************************/

#include "catalog.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

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
 * minimal ini parser: [section] + key=value lines (comments start with #)
 */
std::map<std::string, std::map<std::string, std::string>> parse_ini(const std::string& content) {
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

std::string normalize(const std::string& s) {
    std::string out;
    for (char c : to_lower(s)) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            out.push_back(c);
        }
    }
    return out;
}

std::vector<std::string> section_tokens(const std::string& section) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : to_lower(section)) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cur.push_back(c);
        } else if (!cur.empty()) {
            tokens.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) {
        tokens.push_back(cur);
    }
    // drop generic words
    tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [](const std::string& t) {
        return t == "server" || t == "chat" || t == "cfg" || t == "config";
    }), tokens.end());
    return tokens;
}

std::vector<std::string> list_bin_exes(const std::string& bin_dir) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::exists(bin_dir, ec)) {
        return out;
    }
    for (const auto& entry : fs::directory_iterator(bin_dir, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        auto name = entry.path().filename().string();
        if (name.size() > 4 && name.substr(name.size() - 4) == ".out") {
            out.push_back(name);
        }
    }
    return out;
}

/***
 * match a conf/server config to an executable by section-name token overlap
 */
std::string match_exe(const std::string& section, const std::string& bin_dir) {
    auto tokens = section_tokens(section);
    if (tokens.empty()) {
        return "";
    }
    for (const auto& exe : list_bin_exes(bin_dir)) {
        std::string exe_norm = normalize(exe);
        if (exe_norm.find("benchmark") != std::string::npos ||
            exe_norm.find("client") != std::string::npos ||
            exe_norm.find("proxy") != std::string::npos) {
            continue;
        }
        bool all = true;
        for (const auto& t : tokens) {
            if (exe_norm.find(t) == std::string::npos) {
                all = false;
                break;
            }
        }
        if (all) {
            return exe;
        }
    }
    return "";
}

} // namespace

const ServerEntry* Catalog::find(const std::string& id) const {
    for (const auto& e : _entries) {
        if (e.id == id) {
            return &e;
        }
    }
    return nullptr;
}

bool Catalog::init(const std::string& project_root, const std::string& generated_dir) {
    _entries.clear();
    std::string conf_dir = project_root + "/conf/server";
    std::string bin_dir = project_root + "/_bin";

    std::error_code ec;
    if (!fs::exists(conf_dir, ec)) {
        return false;
    }

    for (const auto& entry : fs::recursive_directory_iterator(conf_dir, ec)) {
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".ini") {
            continue;
        }
        auto cfg_path = entry.path().string();
        std::string content = read_file(cfg_path);
        if (content.find("server_uri") == std::string::npos &&
            content.find("server_url") == std::string::npos) {
            continue;
        }

        auto cfg = parse_ini(content);
        std::string section;
        std::string host = "localhost";
        std::string auth_token;
        int port = 0;
        std::string uri;
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
            if (uit != kv.end()) {
                uri = uit->second;
            } else {
                uit = kv.find("server_url");
                if (uit != kv.end()) {
                    uri = uit->second;
                }
            }
            break;
        }
        if (section.empty() || port <= 0 || uri.empty()) {
            continue;
        }

        std::string exe = match_exe(section, bin_dir);
        if (exe.empty()) {
            continue;
        }
        // guard against mislabeled config files: the config's model dir name
        // must appear in the matched executable name
        {
            fs::path cfg_rel = fs::relative(cfg_path, conf_dir, ec);
            std::string model_dir;
            if (cfg_rel.has_parent_path()) {
                model_dir = cfg_rel.parent_path().filename().string();
            }
            std::string model_norm = normalize(model_dir);
            std::string exe_norm = normalize(exe);
            if (!model_norm.empty() && exe_norm.find(model_norm) == std::string::npos) {
                continue;
            }
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
                category = it->string();
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

    add_missing_server(project_root, generated_dir,
                       "densenet_classification_server.out", "DENSENET_CLASSIFICATION_SERVER",
                       "DENSENET", 9004, "/mortred_ai_server_v1/classification/densenet",
                       "classification", "../conf/model/classification/densenet/densenet121_config.ini");
    add_missing_server(project_root, generated_dir,
                       "real_esrgan_server.out", "REAL_ESRGAN_SERVER",
                       "", 9012, "/mortred_ai_server_v1/enhancement/real_esrgan",
                       "enhancement", "../conf/model/enhancement/real_esrgan/real_esrgan.ini");
    return true;
}

void Catalog::add_missing_server(const std::string& project_root,
                                 const std::string& generated_dir,
                                 const std::string& exe,
                                 const std::string& section,
                                 const std::string& model_section,
                                 int port,
                                 const std::string& uri,
                                 const std::string& category,
                                 const std::string& model_cfg_rel) {
    for (const auto& e : _entries) {
        if (e.exe == exe) {
            return;
        }
    }
    std::error_code ec;
    if (!fs::exists(project_root + "/_bin/" + exe, ec)) {
        return;
    }

    fs::create_directories(generated_dir, ec);
    std::string id = exe.substr(0, exe.size() - 4);
    std::string cfg_name = id + "_cfg.ini";
    std::string cfg_path = generated_dir + "/" + cfg_name;
    std::string content =
        "[" + section + "]\n"
        "port=" + std::to_string(port) + "\n"
        "host=\"localhost\"\n"
        "max_connections=500\n"
        "peer_resp_timeout=15\n"
        "request_size_limit=64\n"
        "# auth_token=\"\"\n"
        "# rate_limit_qps=0\n"
        "compute_threads=-1\n"
        "handler_threads=50\n"
        "worker_nums=1\n"
        "model_run_timeout=-1\n";
    if (model_section.empty()) {
        // real_esrgan style: model_config_file_path lives in the server section
        content += "model_config_file_path=\"" + model_cfg_rel + "\"\n";
    }
    content += "server_uri=\"" + uri + "\"\n";
    if (!model_section.empty()) {
        content += "\n[" + model_section + "]\n"
                   "model_config_file_path=\"" + model_cfg_rel + "\"\n";
    }
    std::ofstream out(cfg_path, std::ios::trunc);
    if (!out.is_open()) {
        return;
    }
    out << content;
    out.close();

    ServerEntry e;
    e.id = id;
    e.name = id;
    e.category = category;
    e.exe = exe;
    e.config = cfg_path;
    e.host = "localhost";
    e.port = port;
    e.uri = uri;
    e.type = section.find("CHAT") != std::string::npos ? "chat" : "image";
    e.generated_config = true;
    _entries.push_back(std::move(e));
}

} // namespace mortred_web
