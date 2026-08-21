/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: auth.cpp
 * Date: 26-8-13
 ************************************************/

#include "auth.h"

#include "common/auth_token.h"

namespace mortred_web {

bool is_loopback_host(const std::string& host) {
    return jinq::common::is_loopback_host(host);
}

std::string bearer_token_of(const std::string& authorization_header) {
    return jinq::common::bearer_token_of(authorization_header);
}

bool constant_time_equals(const std::string& lhs, const std::string& rhs) {
    return jinq::common::constant_time_equals(lhs, rhs);
}

bool is_authorized(const std::string& authorization_header,
                   const std::string& configured_token) {
    return jinq::common::is_bearer_authorized(authorization_header,
                                              configured_token);
}

}  // namespace mortred_web
