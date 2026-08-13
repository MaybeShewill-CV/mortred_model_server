/************************************************
 * Author: Codex
 * File: auth.h
 * Date: 2026-08-13
 ************************************************/

#ifndef MORTRED_WEB_CONSOLE_AUTH_H
#define MORTRED_WEB_CONSOLE_AUTH_H

#include <string>

namespace mortred_web {

/***
 * 判断监听地址是否为回环地址（127.0.0.0/8、::1、localhost）。
 */
bool is_loopback_host(const std::string& host);

/***
 * 从 Authorization 请求头中提取 Bearer Token。
 * 非 Bearer 或无 token 时返回空串。
 */
std::string bearer_token_of(const std::string& authorization_header);

/***
 * 常量时间比较，避免通过响应时间差探测 token。
 */
bool constant_time_equals(const std::string& lhs, const std::string& rhs);

/***
 * 鉴权总入口：
 * - 未配置 token（本地回环模式）时放行；
 * - 配置了 token 时，必须匹配 Authorization 头中的 Bearer Token。
 */
bool is_authorized(const std::string& authorization_header,
                   const std::string& configured_token);

}  // namespace mortred_web

#endif  // MORTRED_WEB_CONSOLE_AUTH_H
