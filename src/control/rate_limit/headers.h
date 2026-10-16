/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: headers.h
* Date: 26-10-16
************************************************/

#ifndef MORTRED_CONTROL_RATE_LIMIT_HEADERS_H
#define MORTRED_CONTROL_RATE_LIMIT_HEADERS_H

#include <cstdint>
#include <string>

namespace mortred {
namespace control {
namespace ratelimit {

/*** Shared 429 RateLimit header contract for every product-path layer.
 * Layers: gateway-gcra | api-key-gcra | api-key-max-inflight | api-key-max-queue | api-key-worker-budget
 * (model FixedWindow product path retired; tests-only FixedWindowRateLimiter remains). */
template <typename Resp>
inline void apply_rate_limit_headers(Resp* resp,
                                     int32_t retry_after_ms,
                                     int limit,
                                     const char* layer,
                                     int remaining = 0) {
    if (resp == nullptr) {
        return;
    }
    int retry_s = (retry_after_ms + 999) / 1000;
    if (retry_s < 1) {
        retry_s = 1;
    }
    resp->add_header_pair("Retry-After", std::to_string(retry_s).c_str());
    if (limit > 0) {
        resp->add_header_pair("RateLimit-Limit", std::to_string(limit).c_str());
        resp->add_header_pair("RateLimit-Remaining", std::to_string(remaining < 0 ? 0 : remaining).c_str());
        resp->add_header_pair("RateLimit-Reset", std::to_string(retry_s).c_str());
    }
    if (layer != nullptr && *layer != '\0') {
        resp->add_header_pair("X-Mortred-RateLimit-Layer", layer);
    }
}

}  // namespace ratelimit
}  // namespace control
}  // namespace mortred

#endif  // MORTRED_CONTROL_RATE_LIMIT_HEADERS_H
