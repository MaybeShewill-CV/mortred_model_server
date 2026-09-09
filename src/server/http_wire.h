/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: http_wire.h
* Date: 26-9-9
************************************************/

// Non-template HTTP primitives and unified-envelope writers. Kept off the
// WORKER / MODEL_OUTPUT template so each catalog output type does not copy
// inet_ntop / header-cursor machine code. reply_async_error stays with the
// /jobs adapter (it is not a UnifiedResponse).

#ifndef MORTRED_SERVER_HTTP_WIRE_H
#define MORTRED_SERVER_HTTP_WIRE_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFHttpServer.h"
#include "workflow/WFTaskFactory.h"

#include "common/response_envelope.h"
#include "common/status_code.h"
#include "server/http_status.h"
#include "server/request_admission.h"

namespace jinq {
namespace server {
using jinq::common::StatusCode;

inline std::string generate_req_id() {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           now.time_since_epoch())
                           .count();
    static std::atomic<uint64_t> seq{0};
    const uint64_t unique = static_cast<uint64_t>(nanos) ^
                            (static_cast<uint64_t>(seq.fetch_add(1)) << 32);
    char buf[32] = {0};
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(unique));
    return std::string(buf);
}

inline std::string peer_ip_of(const WFHttpTask* task) {
    struct sockaddr_storage peer_addr;
    socklen_t addr_len = sizeof(peer_addr);
    if (task->get_peer_addr(reinterpret_cast<struct sockaddr*>(&peer_addr), &addr_len) != 0) {
        return "";
    }
    char ip_buf[INET6_ADDRSTRLEN] = {0};
    if (peer_addr.ss_family == AF_INET) {
        auto* ipv4 = reinterpret_cast<const struct sockaddr_in*>(&peer_addr);
        inet_ntop(AF_INET, &ipv4->sin_addr, ip_buf, sizeof(ip_buf));
    } else if (peer_addr.ss_family == AF_INET6) {
        auto* ipv6 = reinterpret_cast<const struct sockaddr_in6*>(&peer_addr);
        inet_ntop(AF_INET6, &ipv6->sin6_addr, ip_buf, sizeof(ip_buf));
    }
    return std::string(ip_buf);
}

inline std::string header_value_of(const protocol::HttpRequest* req,
                                   const std::string& target_name) {
    protocol::HttpHeaderCursor cursor(req);
    protocol::HttpMessageHeader header;
    std::string target = target_name;
    std::transform(target.begin(), target.end(), target.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    while (cursor.next(&header)) {
        std::string name(static_cast<const char*>(header.name), header.name_len);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (name == target) {
            return std::string(static_cast<const char*>(header.value), header.value_len);
        }
    }
    return "";
}

inline std::string authorization_header_of(const protocol::HttpRequest* req) {
    return header_value_of(req, "authorization");
}

inline bool is_json_content_type(const std::string& content_type) {
    return content_type_kind(content_type) == ContentTypeKind::Json;
}

inline bool is_raw_body_content_type(const std::string& content_type) {
    return content_type_kind(content_type) == ContentTypeKind::RawBody;
}

inline jinq::common::UnifiedResponse status_envelope(
    const std::string& task_id, StatusCode status,
    std::vector<jinq::common::ResponseError> errors = {},
    const std::string& model_name = {}) {
    jinq::common::UnifiedResponse unified;
    unified.task_id = task_id;
    unified.status = jinq::common::to_underlying(status);
    unified.status_str = jinq::common::status_code_to_str(status);
    unified.model_name = model_name;
    unified.errors = std::move(errors);
    return unified;
}

inline jinq::common::UnifiedResponse unified_rejection(
    const std::string& task_id, StatusCode status,
    std::vector<jinq::common::ResponseError> errors) {
    return status_envelope(task_id, status, std::move(errors));
}

inline void reply_unified_json(protocol::HttpResponse* resp,
                               const jinq::common::UnifiedResponse& unified) {
    resp->set_status_code(std::to_string(http_status_of(
                              static_cast<StatusCode>(unified.status))).c_str());
    resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
    resp->add_header_pair("X-Request-ID", unified.task_id.c_str());
    resp->add_header_pair("Cache-Control", "no-store");

    const auto body = jinq::common::envelope::encode(unified);
    resp->append_output_body(body.data(), body.size());
}

inline void reply_unified_json(WFHttpTask* task,
                               const jinq::common::UnifiedResponse& unified) {
    reply_unified_json(task->get_resp(), unified);
}

inline void reply_status(WFHttpTask* task, StatusCode status, const std::string& model_name) {
    reply_unified_json(task, status_envelope("", status, {}, model_name));
}

inline void reply_unauthorized(WFHttpTask* task, const std::string& model_name) {
    task->get_resp()->add_header_pair("WWW-Authenticate", "Bearer realm=\"Mortred\"");
    reply_unified_json(task, status_envelope("", StatusCode::UNAUTHORIZED, {}, model_name));
}

inline void reply_rate_limited(WFHttpTask* task, const std::string& model_name) {
    reply_unified_json(task, status_envelope("", StatusCode::RATE_LIMITED, {}, model_name));
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_HTTP_WIRE_H
