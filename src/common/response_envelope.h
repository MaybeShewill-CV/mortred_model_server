/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: response_envelope.h
 * Date: 26-9-4
 ************************************************/

// Unified response-envelope contract (wire JSON only). Field names, encode
// and decode live here. Clients MUST ignore unknown response fields so the
// envelope can grow without a v2. Payload schemas inside results[].data are
// owned by server/response_serializers.h, not this file.
//
//   { "status": 0, "status_str": "OK", "task_id": "...",
//     "model": {"name": "...", "version": "..."},
//     "results": [{"status": 0, "data": ...}],
//     "server_time_ms": 41.2, "partial": false,
//     "errors": [{"pointer": "...", "message": "..."}] }

#ifndef MORTRED_COMMON_RESPONSE_ENVELOPE_H
#define MORTRED_COMMON_RESPONSE_ENVELOPE_H

#include <string>
#include <string_view>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/request_envelope.h"
#include "common/status_code.h"

namespace mortred {
namespace common {

struct ResponseItem {
    int status = 0;
    rapidjson::Document data;
};

struct ResponseError {
    std::string pointer;
    std::string message;
};

struct UnifiedResponse {
    std::string task_id;
    int status = 0;
    std::string status_str;
    std::string model_name;
    std::string model_version;
    std::vector<ResponseItem> results;
    double server_time_ms = 0.0;
    bool partial = false;
    std::vector<ResponseError> errors;
};

/*** The numeric code and its wire string are one value. Pairing them by hand at
 * every reply site is how `status` and `status_str` drift apart on the wire. */
inline void set_status(UnifiedResponse *resp, StatusCode code) {
    resp->status = to_underlying(code);
    resp->status_str = status_code_to_str(code);
}

namespace envelope {

inline constexpr const char *k_status = "status";
inline constexpr const char *k_status_str = "status_str";
inline constexpr const char *k_task_id = "task_id";
inline constexpr const char *k_model = "model";
inline constexpr const char *k_name = "name";
inline constexpr const char *k_version = "version";
inline constexpr const char *k_results = "results";
inline constexpr const char *k_data = "data";
inline constexpr const char *k_server_time_ms = "server_time_ms";
inline constexpr const char *k_partial = "partial";
inline constexpr const char *k_errors = "errors";
inline constexpr const char *k_pointer = "pointer";
inline constexpr const char *k_message = "message";

inline std::string encode(const UnifiedResponse &resp) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);

    writer.StartObject();
    writer.Key(k_status);
    writer.Int(resp.status);
    writer.Key(k_status_str);
    writer.String(resp.status_str.c_str());
    writer.Key(k_task_id);
    writer.String(resp.task_id.c_str());
    writer.Key(k_model);
    writer.StartObject();
    writer.Key(k_name);
    writer.String(resp.model_name.c_str());
    writer.Key(k_version);
    writer.String(resp.model_version.c_str());
    writer.EndObject();
    writer.Key(k_results);
    writer.StartArray();
    for (const auto &item : resp.results) {
        writer.StartObject();
        writer.Key(k_status);
        writer.Int(item.status);
        writer.Key(k_data);
        if (item.data.IsNull()) {
            writer.Null();
        } else {
            item.data.Accept(writer);
        }
        writer.EndObject();
    }
    writer.EndArray();
    writer.Key(k_server_time_ms);
    writer.Double(resp.server_time_ms);
    writer.Key(k_partial);
    writer.Bool(resp.partial);
    if (!resp.errors.empty()) {
        writer.Key(k_errors);
        writer.StartArray();
        for (const auto &error : resp.errors) {
            writer.StartObject();
            writer.Key(k_pointer);
            writer.String(error.pointer.c_str());
            writer.Key(k_message);
            writer.String(error.message.c_str());
            writer.EndObject();
        }
        writer.EndArray();
    }
    writer.EndObject();
    return buf.GetString();
}

} // namespace envelope

} // namespace common
} // namespace mortred

#endif // MORTRED_COMMON_RESPONSE_ENVELOPE_H
