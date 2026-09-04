/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: http_reply.h
 * Date: 26-9-4
 ************************************************/

// Shared JSON reply helpers for control-plane HTTP handlers. Included only by
// gateway/supervisor *_app.cpp (compiled into those executables), so it does
// not pull Workflow into the workflow-free control shared library.
//
// Management APIs keep {ok, error}. Proxy-path local failures (gateway routing
// / auth, supervisor infer/jobs/pipelines before upstream) use UnifiedResponse.

#ifndef MORTRED_CONTROL_HTTP_REPLY_H
#define MORTRED_CONTROL_HTTP_REPLY_H

#include <string>

#include "workflow/HttpMessage.h"
#include "workflow/WFHttpServer.h"

#include "common/response_envelope.h"
#include "common/status_code.h"

namespace mortred {
namespace control {

inline void reply_json(WFHttpTask *task, int http_status, const std::string &body) {
    auto *resp = task->get_resp();
    resp->set_status_code(std::to_string(http_status).c_str());
    resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
    resp->add_header_pair("Cache-Control", "no-store");
    resp->append_output_body(body.data(), body.size());
}

inline jinq::common::StatusCode status_for_proxy_http(int http_status) {
    using jinq::common::StatusCode;
    switch (http_status) {
    case 400:
        return StatusCode::JSON_DECODE_ERROR;
    case 401:
        return StatusCode::UNAUTHORIZED;
    case 404:
        return StatusCode::NOT_FOUND;
    case 405:
        return StatusCode::METHOD_NOT_ALLOWED;
    case 409:
        return StatusCode::NOT_READY;
    case 422:
        return StatusCode::INVALID_REQUEST_PARAMETER;
    case 502:
        return StatusCode::INTERNAL_ERROR;
    case 503:
        return StatusCode::NOT_READY;
    default:
        return StatusCode::INTERNAL_ERROR;
    }
}

inline void reply_unified_error(WFHttpTask *task, int http_status, jinq::common::StatusCode status,
                                const std::string &message, const std::string &pointer = {}) {
    jinq::common::UnifiedResponse unified;
    unified.status = jinq::common::to_underlying(status);
    unified.status_str = jinq::common::status_code_to_str(status);
    if (!message.empty()) {
        jinq::common::ResponseError err;
        err.pointer = pointer;
        err.message = message;
        unified.errors.push_back(std::move(err));
    }
    reply_json(task, http_status, jinq::common::envelope::encode(unified));
}

inline void reply_unified_error(WFHttpTask *task, int http_status, const std::string &message,
                                const std::string &pointer = {}) {
    reply_unified_error(task, http_status, status_for_proxy_http(http_status), message, pointer);
}

} // namespace control
} // namespace mortred

#endif // MORTRED_CONTROL_HTTP_REPLY_H
