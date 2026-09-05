/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: listen_policy.h
* Date: 26-9-5
************************************************/

#ifndef MORTRED_COMMON_LISTEN_POLICY_H
#define MORTRED_COMMON_LISTEN_POLICY_H

#include <cstdlib>
#include <cstring>
#include <string>

#include "common/auth_token.h"

namespace jinq {
namespace common {

/***
 * How Mortred itself may bind. TLS never lives in these processes.
 *
 *   loopback | edge  (default) — 127.0.0.0/8, ::1, localhost only.
 *   docker                 — 0.0.0.0 allowed; host publish stays the real expose
 *                            (compose maps 127.0.0.1:8080:8080).
 *   unsafe                 — 0.0.0.0 on metal; doctor --strict still fails.
 */
inline std::string mortred_expose_mode() {
    const char* env = std::getenv("MORTRED_EXPOSE");
    if (env == nullptr || *env == '\0') {
        return "loopback";
    }
    std::string mode = env;
    for (char& ch : mode) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return mode;
}

inline bool expose_allows_non_loopback() {
    const std::string mode = mortred_expose_mode();
    return mode == "docker" || mode == "unsafe";
}

/*** refuse wildcard listen unless MORTRED_EXPOSE=docker|unsafe */
inline bool listen_host_permitted(const std::string& host) {
    if (is_loopback_host(host)) {
        return true;
    }
    return expose_allows_non_loopback();
}

}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_LISTEN_POLICY_H
