/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: http_response.h
* Date: 26-8-17
************************************************/

// Unified HTTP JSON response envelope used by all model servers.

#ifndef MORTRED_COMMON_HTTP_RESPONSE_H
#define MORTRED_COMMON_HTTP_RESPONSE_H

#include <string>
#include <vector>

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

/*** One per-item entry of the unified response envelope: the item's own
 * status plus its task payload (null when the item produced no payload,
 * e.g. a failed batch member). */
struct ResponseItem {
    int status = 0;  // StatusCode wire value, stable contract
    rapidjson::Document data;
};

/*** Unified response envelope (request-side counterpart:
 * server/request_envelope.h). Rendered by BaseAiServerImpl from M4 on;
 * the structure lives here so schema tests can pin it before wiring.
 *
 *   { "status": 0, "status_str": "OK", "task_id": "...",
 *     "model": {"name": "...", "version": "..."},
 *     "results": [{"status": 0, "data": ...}],
 *     "server_time_ms": 41.2, "partial": false }
 */
struct UnifiedResponse {
    std::string task_id;
    int status = 0;
    std::string status_str;
    std::string model_name;
    std::string model_version;
    std::vector<ResponseItem> results;
    double server_time_ms = 0.0;
    bool partial = false;
};

inline std::string build_unified_response_body(const UnifiedResponse& resp) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);

    writer.StartObject();
    writer.Key("status");
    writer.Int(resp.status);
    writer.Key("status_str");
    writer.String(resp.status_str.c_str());
    writer.Key("task_id");
    writer.String(resp.task_id.c_str());
    writer.Key("model");
    writer.StartObject();
    writer.Key("name");
    writer.String(resp.model_name.c_str());
    writer.Key("version");
    writer.String(resp.model_version.c_str());
    writer.EndObject();
    writer.Key("results");
    writer.StartArray();
    for (const auto& item : resp.results) {
        writer.StartObject();
        writer.Key("status");
        writer.Int(item.status);
        writer.Key("data");
        if (item.data.IsNull()) {
            writer.Null();
        } else {
            item.data.Accept(writer);
        }
        writer.EndObject();
    }
    writer.EndArray();
    writer.Key("server_time_ms");
    writer.Double(resp.server_time_ms);
    writer.Key("partial");
    writer.Bool(resp.partial);
    writer.EndObject();

    return buf.GetString();
}

}  // namespace common
}  // namespace jinq

#endif  // MORTRED_COMMON_HTTP_RESPONSE_H
