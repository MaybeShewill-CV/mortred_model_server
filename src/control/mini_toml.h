/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mini_toml.h
* Date: 26-8-22
************************************************/

// Minimal TOML subset parser shared by the control plane (catalog +
// mortred.toml). Deliberately dependency-free: [section] headers plus
// key = "value" / key = value lines, '#' comments. Enough for the flat
// server/config schemas; full model configs keep using toml11 in the model
// layer.

#ifndef MORTRED_CONTROL_MINI_TOML_H
#define MORTRED_CONTROL_MINI_TOML_H

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace mortred {
namespace control {
namespace mini_toml {

using Table = std::map<std::string, std::string>;
using Doc = std::map<std::string, Table>;
// 1-based source locations, so callers can name the offending line instead of
// only the section. Kept beside the Doc; the parse structures stay unchanged.
using KeyLines = std::map<std::string, std::map<std::string, int>>;
using SectionLines = std::map<std::string, int>;

inline std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

inline std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

inline Doc parse(const std::string& content, std::vector<std::string>* problems = nullptr,
                 KeyLines* key_lines = nullptr, SectionLines* section_lines = nullptr) {
    Doc doc;
    std::string section;
    std::istringstream ss(content);
    std::string line;
    int line_no = 0;
    while (std::getline(ss, line)) {
        ++line_no;
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }
        if (t.front() == '[' && t.back() == ']') {
            section = trim(t.substr(1, t.size() - 2));
            if (section_lines != nullptr) {
                (*section_lines)[section] = line_no;
            }
            continue;
        }
        const auto eq = t.find('=');
        if (eq == std::string::npos || section.empty()) {
            // A config line the parser will drop. Never silent: a typo'd key
            // used to vanish and the caller's defaults silently took over.
            if (problems != nullptr) {
                const std::string where = "line " + std::to_string(line_no) + ": ";
                problems->push_back(where + (section.empty()
                    ? "key-value line outside any [section] dropped: '" + t + "'"
                    : "line without '=' inside [" + section + "] dropped: '" + t + "'"));
            }
            continue;
        }
        std::string value = trim(t.substr(eq + 1));
        const auto hash = value.find(" #");
        if (hash != std::string::npos) {
            value = trim(value.substr(0, hash));
        }
        const std::string key = trim(t.substr(0, eq));
        doc[section][key] = unquote(value);
        if (key_lines != nullptr) {
            (*key_lines)[section][key] = line_no;
        }
    }
    return doc;
}

inline bool load(const std::string& path, Doc* out, KeyLines* key_lines = nullptr,
                 SectionLines* section_lines = nullptr) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<std::string> problems;
    *out = parse(ss.str(), &problems, key_lines, section_lines);
    for (const auto& problem : problems) {
        std::fprintf(stderr, "[mini_toml] %s: %s\n", path.c_str(), problem.c_str());
    }
    return true;
}

inline int to_int(const std::string& s, int fallback) {
    try {
        return std::stoi(s);
    } catch (...) {
        return fallback;
    }
}

inline bool to_bool(const std::string& s, bool fallback) {
    const std::string v = trim(s);
    if (v == "true" || v == "1") {
        return true;
    }
    if (v == "false" || v == "0") {
        return false;
    }
    return fallback;
}

}  // namespace mini_toml
}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_MINI_TOML_H
