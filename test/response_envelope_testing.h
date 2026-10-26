/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: response_envelope_testing.h
* Date: 26-10-26
************************************************/

// Client-side decode of the unified envelope. Production only encodes responses;
// this decoder exists for tests and tooling that must read one back, so it lives
// here instead of next to the production (encode-only) envelope header.

#ifndef MORTRED_COMMON_RESPONSE_ENVELOPE_TESTING_H
#define MORTRED_COMMON_RESPONSE_ENVELOPE_TESTING_H

#include <string_view>

#include "common/response_envelope.h"

namespace mortred {
namespace common {
namespace envelope {

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
} // namespace common
} // namespace mortred

#endif // MORTRED_COMMON_RESPONSE_ENVELOPE_TESTING_H
