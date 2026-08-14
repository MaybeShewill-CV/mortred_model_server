/************************************************
 * Author: Codex
 * File: auth_token.h
 * Date: 2026-08-13
 ************************************************/

#ifndef MORTRED_COMMON_AUTH_TOKEN_H
#define MORTRED_COMMON_AUTH_TOKEN_H

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>

namespace jinq {
namespace common {

/***
 * 判断监听地址是否为回环地址（127.0.0.0/8、::1、localhost）。
 */
inline bool is_loopback_host(const std::string& host) {
    std::string lower_host = host;
    std::transform(lower_host.begin(), lower_host.end(), lower_host.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower_host == "localhost" || lower_host == "::1" || lower_host == "[::1]") {
        return true;
    }
    // strict 127.0.0.0/8 check: "127." followed by a dotted quad (digits and
    // exactly two more dots), rejecting hostnames like "127.evil"
    if (lower_host.compare(0, 4, "127.") != 0) {
        return false;
    }
    int dots = 0;
    for (size_t i = 4; i < lower_host.size(); ++i) {
        const char ch = lower_host[i];
        if (ch == '.') {
            ++dots;
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    return dots == 2;
}

/***
 * 从 Authorization 请求头中提取 Bearer Token。
 * 非 Bearer 或无 token 时返回空串。
 */
inline std::string bearer_token_of(const std::string& authorization_header) {
    const std::string scheme = "bearer ";
    std::string header = authorization_header;
    size_t b = 0;
    size_t e = header.size();
    while (b < e && std::isspace(static_cast<unsigned char>(header[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(header[e - 1]))) {
        --e;
    }
    header = header.substr(b, e - b);

    std::string lower_header = header;
    std::transform(lower_header.begin(), lower_header.end(), lower_header.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower_header.size() <= scheme.size() ||
        lower_header.compare(0, scheme.size(), scheme) != 0) {
        return "";
    }
    std::string token = header.substr(scheme.size());
    b = 0;
    e = token.size();
    while (b < e && std::isspace(static_cast<unsigned char>(token[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(token[e - 1]))) {
        --e;
    }
    return token.substr(b, e - b);
}

/***
 * 常量时间比较，避免通过响应时间差探测 token。
 * 长度差异被合并进掩码，比较时长只与较长的输入有关，不泄漏长度信息。
 */
inline bool constant_time_equals(const std::string& lhs, const std::string& rhs) {
    const size_t n = std::max(lhs.size(), rhs.size());
    unsigned char diff = static_cast<unsigned char>(lhs.size() ^ rhs.size());
    for (size_t i = 0; i < n; ++i) {
        const unsigned char a = i < lhs.size() ? static_cast<unsigned char>(lhs[i]) : 0;
        const unsigned char b = i < rhs.size() ? static_cast<unsigned char>(rhs[i]) : 0;
        diff |= static_cast<unsigned char>(a ^ b);
    }
    return diff == 0;
}

/***
 * 鉴权总入口：
 * - 未配置 token（本地回环模式）时放行；
 * - 配置了 token 时，必须匹配 Authorization 头中的 Bearer Token。
 */
inline bool is_bearer_authorized(const std::string& authorization_header,
                                 const std::string& configured_token) {
    if (configured_token.empty()) {
        return true;
    }
    return constant_time_equals(bearer_token_of(authorization_header),
                                configured_token);
}

}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_AUTH_TOKEN_H
