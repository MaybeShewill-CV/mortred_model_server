/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: request_envelope.h
 * Date: 26-8-31
 ************************************************/

// Unified request envelope parser. The envelope is the fixed outer structure
// of the JSON body (identity + payload slots + request-level metadata):
//
//   { "req_id": "...",            optional trace id, echoed as task_id
//     "images": ["<base64>",..],  required, >=1, always an array
//     "params":  { ... },         optional, validated against the model's
//                                  ParamSpec schema (strict: unknown = 422)
//     "options": { ... } }        optional output options
//
// Naming symmetry: http_response.h documents the response envelope; this
// header is its request-side counterpart. "message" is deliberately avoided:
// an HTTP request message is the whole request (line + headers + body), this
// header defines only the body's logical wrapper.

#ifndef MORTRED_SERVER_REQUEST_ENVELOPE_H
#define MORTRED_SERVER_REQUEST_ENVELOPE_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rapidjson/document.h"

#include "common/status_code.h"
#include "models/backend/param_spec.h"
#include "models/io/common_input.h"
#include "server/output_options.h"

namespace jinq {
namespace server {

using jinq::models::backend::ParamSet;
using jinq::models::backend::ParamSpec;
using jinq::models::backend::ParamValue;
using jinq::models::backend::ParamViolation;
using jinq::models::io_define::common_io::byte_source;
using jinq::common::StatusCode;

struct ParsedRequest {
    std::string req_id;                    // may stay empty: server generates
    std::vector<byte_source> items;        // one entry per images[] element
    std::shared_ptr<ParamSet> params;      // nullptr when the params key is absent
    OutputOptions options;                 // defaults when the options key is absent

    bool is_valid = false;
    StatusCode status = StatusCode::OK;    // coarse first-failure status
    std::vector<ParamViolation> violations; // strict errors with JSON pointers
};

namespace detail {

inline const char *k_allowed_envelope_keys = "req_id, images, params, options";

inline const char *k_img_data_migration =
    "field 'img_data' was removed; use images: [\"<base64>\"] (migration: img_data -> images[0])";

/*** maps a rapidjson scalar member into a JSON-typed ParamValue; returns
 * false when the value is not a scalar (null / array / object) */
inline bool param_value_of(const rapidjson::Value &value, ParamValue *out) {
    if (value.IsBool()) {
        *out = ParamValue::of(value.GetBool());
    } else if (value.IsInt64()) {
        *out = ParamValue::of(value.GetInt64());
    } else if (value.IsUint64()) {
        *out = ParamValue::of(static_cast<int64_t>(value.GetUint64()));
    } else if (value.IsDouble()) {
        *out = ParamValue::of(value.GetDouble());
    } else if (value.IsString()) {
        *out = ParamValue::of(std::string(value.GetString(), value.GetStringLength()));
    } else {
        return false;
    }
    return true;
}

} // namespace detail

/*** Strict parse of the unified request envelope.
 *
 * Never throws; every rejection carries a JSON pointer and a message:
 *   - malformed JSON / non-object body      -> JSON_DECODE_ERROR, "/"
 *   - legacy "img_data"                     -> INVALID_REQUEST_PARAMETER + migration hint
 *   - unknown envelope/param/option keys    -> pointer + allowed-key list
 *   - images shape errors                   -> "/images" or "/images/<index>"
 *   - param type/range/enum/duplicate       -> "/params/<key>"
 *
 * items/params/options are only meaningful when is_valid is true.
 */
inline ParsedRequest parse_request_envelope(const std::string &body, const std::vector<ParamSpec> &specs) {
    ParsedRequest request;

    rapidjson::Document doc;
    if (body.empty()) {
        doc.Parse("{}", 2);
        request.status = StatusCode::JSON_DECODE_ERROR;
        request.violations.push_back({"/", "request body is empty; expected a JSON object"});
        return request;
    }
    doc.Parse(body.data(), body.size());
    if (doc.HasParseError() || !doc.IsObject()) {
        request.status = StatusCode::JSON_DECODE_ERROR;
        request.violations.push_back({"/", "request body must be a JSON object"});
        return request;
    }

    // the legacy field never succeeds: fail fast with exactly one
    // actionable migration message, before any other shape complaint
    if (doc.HasMember("img_data")) {
        request.violations.push_back({"/img_data", detail::k_img_data_migration});
        request.is_valid = false;
        request.status = StatusCode::INVALID_REQUEST_PARAMETER;
        return request;
    }

    for (auto member = doc.MemberBegin(); member != doc.MemberEnd(); ++member) {
        const std::string key(member->name.GetString(), member->name.GetStringLength());
        if (key == "req_id" || key == "images" || key == "params" || key == "options") {
            continue;
        }
        request.violations.push_back(
            {"/" + key, "unknown request field '" + key + "'; allowed: " + detail::k_allowed_envelope_keys});
    }

    if (doc.HasMember("req_id")) {
        const rapidjson::Value &value = doc["req_id"];
        if (!value.IsString()) {
            request.violations.push_back({"/req_id", "req_id must be a string"});
        } else {
            request.req_id.assign(value.GetString(), value.GetStringLength());
        }
    }

    if (!doc.HasMember("images")) {
        request.violations.push_back({"/images", "required field 'images' is missing"});
    } else if (!doc["images"].IsArray()) {
        request.violations.push_back({"/images", "images must be an array of base64 strings"});
    } else {
        const rapidjson::Value &images = doc["images"];
        if (images.Empty()) {
            request.violations.push_back({"/images", "images must contain at least one image"});
        }
        for (rapidjson::SizeType index = 0; index < images.Size(); ++index) {
            const rapidjson::Value &element = images[index];
            const std::string pointer = "/images/" + std::to_string(index);
            if (!element.IsString()) {
                request.violations.push_back({pointer, "images[" + std::to_string(index) + "] must be a string"});
                continue;
            }
            if (element.GetStringLength() == 0) {
                request.violations.push_back(
                    {pointer, "images[" + std::to_string(index) + "] must be a non-empty base64 string"});
                continue;
            }
            byte_source item;
            item.origin = byte_source::origin_kind::base64_text;
            item.data.assign(element.GetString(), element.GetStringLength());
            request.items.push_back(std::move(item));
        }
    }

    if (doc.HasMember("params")) {
        const rapidjson::Value &params = doc["params"];
        if (!params.IsObject()) {
            request.violations.push_back({"/params", "params must be an object"});
        } else {
            std::vector<std::pair<std::string, ParamValue>> candidates;
            bool params_clean = true;
            for (auto member = params.MemberBegin(); member != params.MemberEnd(); ++member) {
                const std::string key(member->name.GetString(), member->name.GetStringLength());
                ParamValue value;
                if (!detail::param_value_of(member->value, &value)) {
                    request.violations.push_back(
                        {"/params/" + key, "parameter value must be a scalar (number, boolean or string)"});
                    params_clean = false;
                    continue;
                }
                candidates.emplace_back(key, std::move(value));
            }
            ParamSet validated;
            const auto param_violations = jinq::models::backend::validate_params(specs, candidates, &validated);
            for (const auto &violation : param_violations) {
                request.violations.push_back(
                    {violation.pointer == "/" ? "/params" : "/params" + violation.pointer, violation.message});
            }
            if (params_clean && param_violations.empty()) {
                request.params = std::make_shared<ParamSet>(std::move(validated));
            }
        }
    }

    if (doc.HasMember("options")) {
        const rapidjson::Value &options = doc["options"];
        if (!options.IsObject()) {
            request.violations.push_back({"/options", "options must be an object"});
        } else {
            OutputOptions parsed;
            const auto option_violations = parse_output_options(options, &parsed);
            for (const auto &violation : option_violations) {
                request.violations.push_back({"/options" + violation.pointer, violation.message});
            }
            if (option_violations.empty()) {
                request.options = parsed;
            }
        }
    }

    request.is_valid = request.violations.empty();
    request.status = request.is_valid ? StatusCode::OK : StatusCode::INVALID_REQUEST_PARAMETER;
    return request;
}

} // namespace server
} // namespace jinq

#endif // MORTRED_SERVER_REQUEST_ENVELOPE_H
