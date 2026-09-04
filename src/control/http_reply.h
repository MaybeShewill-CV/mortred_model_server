/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: http_reply.h
 * Date: 26-9-4
 ************************************************/

// Shared JSON reply helper for control-plane HTTP handlers. Included only by
// gateway/supervisor *_app.cpp (compiled into those executables), so it does
// not pull Workflow into the workflow-free control shared library. Gateway
// and supervisor keep their own error JSON shapes.

#ifndef MORTRED_CONTROL_HTTP_REPLY_H
#define MORTRED_CONTROL_HTTP_REPLY_H

#include <string>

#include "workflow/HttpMessage.h"
#include "workflow/WFHttpServer.h"

namespace mortred {
namespace control {

inline void reply_json(WFHttpTask *task, int http_status, const std::string &body) {
    auto *resp = task->get_resp();
    resp->set_status_code(std::to_string(http_status).c_str());
    resp->add_header_pair("Content-Type", "application/json; charset=utf-8");
    resp->add_header_pair("Cache-Control", "no-store");
    resp->append_output_body(body.data(), body.size());
}

} // namespace control
} // namespace mortred

#endif // MORTRED_CONTROL_HTTP_REPLY_H
