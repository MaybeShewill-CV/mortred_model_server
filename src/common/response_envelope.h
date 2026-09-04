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

namespace jinq {
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

inline DecodeResult<UnifiedResponse> decode_response(std::string_view body) {
    DecodeResult<UnifiedResponse> out;
    if (body.empty()) {
        out.status = StatusCode::JSON_DECODE_ERROR;
        out.violations.push_back({"/", "response body is empty; expected a JSON object"});
        return out;
    }
    rapidjson::Document doc;
    doc.Parse(body.data(), body.size());
    if (doc.HasParseError() || !doc.IsObject()) {
        out.status = StatusCode::JSON_DECODE_ERROR;
        out.violations.push_back({"/", "response body must be a JSON object"});
        return out;
    }

    // unknown keys are ignored (forward-compatible clients)
    if (doc.HasMember(k_status) && doc[k_status].IsInt()) {
        out.value.status = doc[k_status].GetInt();
    }
    if (doc.HasMember(k_status_str) && doc[k_status_str].IsString()) {
        out.value.status_str.assign(doc[k_status_str].GetString(), doc[k_status_str].GetStringLength());
    }
    if (doc.HasMember(k_task_id) && doc[k_task_id].IsString()) {
        out.value.task_id.assign(doc[k_task_id].GetString(), doc[k_task_id].GetStringLength());
    }
    if (doc.HasMember(k_model) && doc[k_model].IsObject()) {
        const auto &model = doc[k_model];
        if (model.HasMember(k_name) && model[k_name].IsString()) {
            out.value.model_name.assign(model[k_name].GetString(), model[k_name].GetStringLength());
        }
        if (model.HasMember(k_version) && model[k_version].IsString()) {
            out.value.model_version.assign(model[k_version].GetString(),
                                           model[k_version].GetStringLength());
        }
    }
    if (doc.HasMember(k_server_time_ms) && doc[k_server_time_ms].IsNumber()) {
        out.value.server_time_ms = doc[k_server_time_ms].GetDouble();
    }
    if (doc.HasMember(k_partial) && doc[k_partial].IsBool()) {
        out.value.partial = doc[k_partial].GetBool();
    }
    if (doc.HasMember(k_results)) {
        if (!doc[k_results].IsArray()) {
            out.status = StatusCode::INVALID_REQUEST_PARAMETER;
            out.violations.push_back({"/results", "results must be an array"});
            return out;
        }
        const auto &results = doc[k_results];
        out.value.results.reserve(results.Size());
        for (rapidjson::SizeType index = 0; index < results.Size(); ++index) {
            const auto &element = results[index];
            ResponseItem item;
            if (element.IsObject()) {
                if (element.HasMember(k_status) && element[k_status].IsInt()) {
                    item.status = element[k_status].GetInt();
                }
                if (element.HasMember(k_data) && !element[k_data].IsNull()) {
                    copy_value(element[k_data], &item.data);
                }
            }
            out.value.results.push_back(std::move(item));
        }
    }
    if (doc.HasMember(k_errors) && doc[k_errors].IsArray()) {
        for (const auto &element : doc[k_errors].GetArray()) {
            if (!element.IsObject()) {
                continue;
            }
            ResponseError error;
            if (element.HasMember(k_pointer) && element[k_pointer].IsString()) {
                error.pointer.assign(element[k_pointer].GetString(),
                                     element[k_pointer].GetStringLength());
            }
            if (element.HasMember(k_message) && element[k_message].IsString()) {
                error.message.assign(element[k_message].GetString(),
                                     element[k_message].GetStringLength());
            }
            out.value.errors.push_back(std::move(error));
        }
    }

    out.ok = true;
    out.status = StatusCode::OK;
    return out;
}

} // namespace envelope

inline std::string build_unified_response_body(const UnifiedResponse &resp) {
    return envelope::encode(resp);
}

} // namespace common
} // namespace jinq

#endif // MORTRED_COMMON_RESPONSE_ENVELOPE_H
