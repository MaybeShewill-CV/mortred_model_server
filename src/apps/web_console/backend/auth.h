/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: auth.h
* Date: 26-8-13
************************************************/

#ifndef MORTRED_WEB_CONSOLE_AUTH_H
#define MORTRED_WEB_CONSOLE_AUTH_H

#include <string>

namespace mortred_web {

/***
 * Whether the listen address is a loopback host (127.0.0.0/8, ::1, localhost).
 */
bool is_loopback_host(const std::string& host);

/***
 * Extract the Bearer Token from the Authorization header.
 * Returns "" when the header is not Bearer or has no token.
 */
std::string bearer_token_of(const std::string& authorization_header);

/***
 * Constant-time comparison to prevent token probing via response timing.
 */
bool constant_time_equals(const std::string& lhs, const std::string& rhs);

/***
 * Auth entry point:
 * - no token configured (loopback mode) -> allow;
 * - token configured -> must match the Bearer Token in the Authorization header.
 */
bool is_authorized(const std::string& authorization_header,
                   const std::string& configured_token);

}  // namespace mortred_web

#endif  // MORTRED_WEB_CONSOLE_AUTH_H
