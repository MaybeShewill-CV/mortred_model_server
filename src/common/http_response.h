/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: http_response.h
* Date: 26-8-17
************************************************/

// Process-level JSON body for non-inference HTTP exits (health, 401, 429,
// 404, 415, ...): {req_id, code, msg, data}. The unified inference response
// envelope is common/response_envelope.h (UnifiedResponse).

#ifndef MORTRED_COMMON_HTTP_RESPONSE_H
#define MORTRED_COMMON_HTTP_RESPONSE_H

#include <string>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/response_envelope.h"

namespace jinq {
namespace common {

struct HttpResponse {
    std::string req_id;
    int code = 0;
    std::string msg;
    rapidjson::Document data;
};

inline std::string build_response_body(const HttpResponse& resp) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);

    writer.StartObject();
    writer.Key("req_id");
    writer.String(resp.req_id.c_str());
    writer.Key("code");
    writer.Int(resp.code);
    writer.Key("msg");
    writer.String(resp.msg.c_str());
    writer.Key("data");
    if (resp.data.IsNull()) {
        writer.Null();
    } else {
        resp.data.Accept(writer);
    }
    writer.EndObject();

    return buf.GetString();
}

}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_HTTP_RESPONSE_H
