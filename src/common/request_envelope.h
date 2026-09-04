/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: request_envelope.h
 * Date: 26-9-4
 ************************************************/

// Unified request-envelope contract (wire JSON only). Field names, encode
// and decode live here and nowhere else. This header is opencv/workflow-free
// so the control plane and tests-only CI can share it.
//
//   { "req_id": "...",            optional; omitted when empty
//     "images": ["<base64>",..],  required, >=1, always an array
//     "params":  { ... },         optional object (schema applied by the
//                                 data-plane binder, not here)
//     "options": { ... } }        optional object (same)
//
// ParamSpec / OutputOptions / byte_source binding belongs in
// server/parsed_request.h. Do not add a second request_envelope.h.

#ifndef MORTRED_COMMON_REQUEST_ENVELOPE_H
#define MORTRED_COMMON_REQUEST_ENVELOPE_H

#include <string>
#include <string_view>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/status_code.h"

namespace jinq {
namespace common {
namespace envelope {

using jinq::common::StatusCode;

inline constexpr const char *k_req_id = "req_id";
inline constexpr const char *k_images = "images";
inline constexpr const char *k_params = "params";
inline constexpr const char *k_options = "options";
inline constexpr const char *k_img_data = "img_data";

inline const char *k_allowed_request_keys = "req_id, images, params, options";

inline const char *k_img_data_migration =
    "field 'img_data' was removed; use images: [\"<base64>\"] (migration: img_data -> images[0])";

struct EnvelopeError {
    std::string pointer;
    std::string message;
};

struct Request {
    std::string req_id;
    std::vector<std::string> images;
    bool has_params = false;
    bool has_options = false;
    rapidjson::Document params;
    rapidjson::Document options;
};

template <typename T>
struct DecodeResult {
    bool ok = false;
    StatusCode status = StatusCode::OK;
    std::vector<EnvelopeError> violations;
    T value;
};

inline void copy_value(const rapidjson::Value &from, rapidjson::Document *to) {
    to->CopyFrom(from, to->GetAllocator());
}

inline DecodeResult<Request> decode_request(const rapidjson::Value &doc) {
    DecodeResult<Request> out;
    if (!doc.IsObject()) {
        out.status = StatusCode::JSON_DECODE_ERROR;
        out.violations.push_back({"/", "request body must be a JSON object"});
        return out;
    }

    if (doc.HasMember(k_img_data)) {
        out.violations.push_back({"/img_data", k_img_data_migration});
        out.status = StatusCode::INVALID_REQUEST_PARAMETER;
        return out;
    }

    for (auto member = doc.MemberBegin(); member != doc.MemberEnd(); ++member) {
        const std::string key(member->name.GetString(), member->name.GetStringLength());
        if (key == k_req_id || key == k_images || key == k_params || key == k_options) {
            continue;
        }
        out.violations.push_back(
            {"/" + key, "unknown request field '" + key + "'; allowed: " + k_allowed_request_keys});
    }

    if (doc.HasMember(k_req_id)) {
        const rapidjson::Value &value = doc[k_req_id];
        if (!value.IsString()) {
            out.violations.push_back({"/req_id", "req_id must be a string"});
        } else {
            out.value.req_id.assign(value.GetString(), value.GetStringLength());
        }
    }

    if (!doc.HasMember(k_images)) {
        out.violations.push_back({"/images", "required field 'images' is missing"});
    } else if (!doc[k_images].IsArray()) {
        out.violations.push_back({"/images", "images must be an array of base64 strings"});
    } else {
        const rapidjson::Value &images = doc[k_images];
        if (images.Empty()) {
            out.violations.push_back({"/images", "images must contain at least one image"});
        }
        for (rapidjson::SizeType index = 0; index < images.Size(); ++index) {
            const rapidjson::Value &element = images[index];
            const std::string pointer = "/images/" + std::to_string(index);
            if (!element.IsString()) {
                out.violations.push_back(
                    {pointer, "images[" + std::to_string(index) + "] must be a string"});
                continue;
            }
            if (element.GetStringLength() == 0) {
                out.violations.push_back(
                    {pointer, "images[" + std::to_string(index) + "] must be a non-empty base64 string"});
                continue;
            }
            out.value.images.emplace_back(element.GetString(), element.GetStringLength());
        }
    }

    if (doc.HasMember(k_params)) {
        const rapidjson::Value &params = doc[k_params];
        if (!params.IsObject()) {
            out.violations.push_back({"/params", "params must be an object"});
        } else {
            out.value.has_params = true;
            copy_value(params, &out.value.params);
        }
    }

    if (doc.HasMember(k_options)) {
        const rapidjson::Value &options = doc[k_options];
        if (!options.IsObject()) {
            out.violations.push_back({"/options", "options must be an object"});
        } else {
            out.value.has_options = true;
            copy_value(options, &out.value.options);
        }
    }

    out.ok = out.violations.empty();
    out.status = out.ok ? StatusCode::OK : StatusCode::INVALID_REQUEST_PARAMETER;
    return out;
}

inline DecodeResult<Request> decode_request(std::string_view body) {
    DecodeResult<Request> out;
    if (body.empty()) {
        out.status = StatusCode::JSON_DECODE_ERROR;
        out.violations.push_back({"/", "request body is empty; expected a JSON object"});
        return out;
    }
    rapidjson::Document doc;
    doc.Parse(body.data(), body.size());
    if (doc.HasParseError() || !doc.IsObject()) {
        out.status = StatusCode::JSON_DECODE_ERROR;
        out.violations.push_back({"/", "request body must be a JSON object"});
        return out;
    }
    return decode_request(doc);
}

inline std::string encode(const Request &request) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();
    if (!request.req_id.empty()) {
        writer.Key(k_req_id);
        writer.String(request.req_id.c_str(), static_cast<rapidjson::SizeType>(request.req_id.size()));
    }
    writer.Key(k_images);
    writer.StartArray();
    for (const auto &image : request.images) {
        writer.String(image.c_str(), static_cast<rapidjson::SizeType>(image.size()));
    }
    writer.EndArray();
    if (request.has_params) {
        writer.Key(k_params);
        request.params.Accept(writer);
    }
    if (request.has_options) {
        writer.Key(k_options);
        request.options.Accept(writer);
    }
    writer.EndObject();
    return buf.GetString();
}

} // namespace envelope
} // namespace common
} // namespace jinq

#endif // MORTRED_COMMON_REQUEST_ENVELOPE_H
