/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: management_envelope.h
 * Date: 26-9-4
 ************************************************/

// Management-plane wrapper around the unified data-plane envelope.
// Infer/jobs/pipelines add server_id, model, steps; the payload is always
// encode/decode from common/request_envelope.h and common/response_envelope.h.

#ifndef MORTRED_CONTROL_MANAGEMENT_ENVELOPE_H
#define MORTRED_CONTROL_MANAGEMENT_ENVELOPE_H

#include <string>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/request_envelope.h"
#include "common/response_envelope.h"
#include "common/status_code.h"

namespace mortred {
namespace control {

using jinq::common::StatusCode;
using jinq::common::UnifiedResponse;
using jinq::common::envelope::Request;
using jinq::common::envelope::decode_request;
using jinq::common::envelope::decode_response;
using jinq::common::envelope::encode;
using jinq::common::envelope::k_images;
using jinq::common::envelope::k_img_data;
using jinq::common::envelope::k_img_data_migration;
using jinq::common::envelope::k_options;
using jinq::common::envelope::k_params;
using jinq::common::envelope::k_req_id;

// Re-export so existing tests can lock the migration text through this header.
// The string is defined only in common/request_envelope.h.

struct EnvelopeRewrite {
    bool ok = false;
    int http_status = 400;
    std::string pointer;
    std::string message;
    std::string json;
};

inline int http_status_of_decode(StatusCode status) {
    return status == StatusCode::JSON_DECODE_ERROR ? 400 : 422;
}

inline rapidjson::Document project_request_object(const rapidjson::Value &doc) {
    rapidjson::Document next;
    next.SetObject();
    auto &allocator = next.GetAllocator();
    static const char *k_keys[] = {k_req_id, k_images, k_params, k_options};
    for (const char *key : k_keys) {
        if (doc.HasMember(key)) {
            next.AddMember(rapidjson::Value(key, allocator), rapidjson::Value(doc[key], allocator),
                           allocator);
        }
    }
    return next;
}

inline EnvelopeRewrite from_decode(const jinq::common::envelope::DecodeResult<Request> &decoded) {
    EnvelopeRewrite out;
    if (!decoded.ok) {
        out.http_status = http_status_of_decode(decoded.status);
        if (!decoded.violations.empty()) {
            out.pointer = decoded.violations[0].pointer;
            out.message = decoded.violations[0].message;
        }
        return out;
    }
    out.ok = true;
    out.http_status = 200;
    out.json = encode(decoded.value);
    return out;
}

/*** Drop management-only fields, then decode+encode the data-plane envelope.
 * img_data is a hard 422 even when images[] is also present. */
inline EnvelopeRewrite copy_request_envelope(const rapidjson::Value &doc) {
    EnvelopeRewrite out;
    if (!doc.IsObject()) {
        out.pointer = "/";
        out.message = "request body must be a JSON object";
        return out;
    }
    if (doc.HasMember(k_img_data)) {
        out.http_status = 422;
        out.pointer = "/img_data";
        out.message = k_img_data_migration;
        return out;
    }
    const auto stripped = project_request_object(doc);
    return from_decode(decode_request(stripped));
}

inline EnvelopeRewrite extract_prev_output_images(const std::string &prev_json,
                                                  const std::string &field) {
    EnvelopeRewrite out;
    const std::string missing = "cannot extract '" + field +
                                "' from previous output; expected unified results[].data";
    const auto decoded = decode_response(prev_json);
    if (!decoded.ok || decoded.value.results.empty()) {
        out.message = missing;
        return out;
    }

    auto &item = decoded.value.results[0];
    if (item.data.IsNull() || !item.data.IsObject() || !item.data.HasMember(field.c_str())) {
        if (item.data.IsNull() || !item.data.IsObject()) {
            out.message = "cannot extract '" + field +
                          "' from previous output; results[0].data is missing";
        } else {
            out.message = "cannot extract '" + field + "' from previous output";
        }
        return out;
    }

    const rapidjson::Value &value = item.data[field.c_str()];
    Request next;
    if (value.IsString() && value.GetStringLength() > 0) {
        next.images.emplace_back(value.GetString(), value.GetStringLength());
    } else if (value.IsArray() && !value.Empty()) {
        for (const auto &element : value.GetArray()) {
            if (!element.IsString() || element.GetStringLength() == 0) {
                out.message = "cannot extract '" + field +
                              "': expected a base64 string or array of strings";
                return out;
            }
            next.images.emplace_back(element.GetString(), element.GetStringLength());
        }
    } else {
        out.message =
            "cannot extract '" + field + "': expected a base64 string or array of strings";
        return out;
    }

    out.ok = true;
    out.http_status = 200;
    out.json = encode(next);
    return out;
}

inline EnvelopeRewrite apply_pipeline_step_input(const std::string &current_body,
                                                 const std::string &input_key) {
    if (input_key.rfind("prev_output.", 0) == 0) {
        return extract_prev_output_images(current_body, input_key.substr(12));
    }
    EnvelopeRewrite out;
    out.ok = true;
    out.http_status = 200;
    out.json = current_body;
    return out;
}

inline std::string encode_infer_proxy(const std::string &server_id, const Request &request) {
    rapidjson::Document doc;
    doc.Parse(encode(request).c_str());
    auto &allocator = doc.GetAllocator();
    doc.AddMember("server_id", rapidjson::Value(server_id.c_str(), server_id.size(), allocator),
                  allocator);
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    doc.Accept(writer);
    return buf.GetString();
}

} // namespace control
} // namespace mortred

#endif // MORTRED_CONTROL_MANAGEMENT_ENVELOPE_H
