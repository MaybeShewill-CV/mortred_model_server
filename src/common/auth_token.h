/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: auth_token.h
* Date: 26-8-13
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
 * Whether the listen address is a loopback host (127.0.0.0/8, ::1, localhost).
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
 * Extract the Bearer Token from the Authorization header.
 * Returns "" when the header is not Bearer or has no token.
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
 * Constant-time comparison to prevent token probing via response timing.
 * Length difference is folded into the mask; runtime depends only on the
 * longer input, so no length information leaks.
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
 * Auth entry point:
 * - no token configured (loopback mode) -> allow;
 * - token configured -> must match the Bearer Token in the Authorization header.
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
