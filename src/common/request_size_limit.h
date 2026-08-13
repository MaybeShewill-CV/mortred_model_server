/************************************************
 * Author: Codex
 * File: request_size_limit.h
 * Date: 2026-08-13
 ************************************************/

#ifndef MORTRED_COMMON_REQUEST_SIZE_LIMIT_H
#define MORTRED_COMMON_REQUEST_SIZE_LIMIT_H

#include <cstddef>
#include <cstdint>

namespace jinq {
namespace common {

/***
 * 默认请求体大小上限（单位 MB）。
 * 所有 HTTP 服务（模型 server / web console）统一以此为默认值。
 */
inline constexpr size_t k_default_request_size_limit_mb = 64;

}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_REQUEST_SIZE_LIMIT_H
