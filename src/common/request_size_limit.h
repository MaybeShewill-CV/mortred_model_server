/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: request_size_limit.h
* Date: 26-8-13
************************************************/

#ifndef MORTRED_COMMON_REQUEST_SIZE_LIMIT_H
#define MORTRED_COMMON_REQUEST_SIZE_LIMIT_H

#include <cstddef>
#include <cstdint>

namespace jinq {
namespace common {

/***
 * Default request body size limit (MB).
 * Used by all HTTP services (model server / web console).
 */
inline constexpr size_t k_default_request_size_limit_mb = 64;

}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_REQUEST_SIZE_LIMIT_H
