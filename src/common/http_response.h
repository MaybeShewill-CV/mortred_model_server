/************************************************
 * Author: Codex
 * File: http_response.h
 * Date: 2026-08-26
 *
 * Unified HTTP JSON response envelope used by all model servers.
 ************************************************/

#ifndef MORTRED_COMMON_HTTP_RESPONSE_H
#define MORTRED_COMMON_HTTP_RESPONSE_H

#include <string>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

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
