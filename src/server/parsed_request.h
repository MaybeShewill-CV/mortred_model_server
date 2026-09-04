/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: parsed_request.h
 * Date: 26-9-4
 ************************************************/

// Data-plane binding of the unified request envelope: wire JSON/raw →
// ParsedRequest (byte_source items, ParamSet, OutputOptions). The envelope
// contract itself is common/request_envelope.h; this header must not
// redefine field names or the img_data migration text.

#ifndef MORTRED_SERVER_PARSED_REQUEST_H
#define MORTRED_SERVER_PARSED_REQUEST_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rapidjson/document.h"

#include "common/request_envelope.h"
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
using jinq::common::envelope::DecodeResult;
using jinq::common::envelope::Request;

struct ParsedRequest {
    std::string req_id;
    std::vector<byte_source> items;
    std::shared_ptr<ParamSet> params;
    OutputOptions options;

    bool is_valid = false;
    StatusCode status = StatusCode::OK;
    std::vector<ParamViolation> violations;
};

namespace detail {

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

inline void append_decode_violations(const DecodeResult<Request> &decoded, ParsedRequest *request) {
    for (const auto &violation : decoded.violations) {
        request->violations.push_back({violation.pointer, violation.message});
    }
}

inline void bind_params(const rapidjson::Value &params, const std::vector<ParamSpec> &specs,
                        ParsedRequest *request) {
    std::vector<std::pair<std::string, ParamValue>> candidates;
    bool params_clean = true;
    for (auto member = params.MemberBegin(); member != params.MemberEnd(); ++member) {
        const std::string key(member->name.GetString(), member->name.GetStringLength());
        ParamValue value;
        if (!param_value_of(member->value, &value)) {
            request->violations.push_back(
                {"/params/" + key, "parameter value must be a scalar (number, boolean or string)"});
            params_clean = false;
            continue;
        }
        candidates.emplace_back(key, std::move(value));
    }
    ParamSet validated;
    const auto param_violations = jinq::models::backend::validate_params(specs, candidates, &validated);
    for (const auto &violation : param_violations) {
        request->violations.push_back(
            {violation.pointer == "/" ? "/params" : "/params" + violation.pointer, violation.message});
    }
    if (params_clean && param_violations.empty()) {
        request->params = std::make_shared<ParamSet>(std::move(validated));
    }
}

inline void bind_options(const rapidjson::Value &options, ParsedRequest *request) {
    OutputOptions parsed;
    const auto option_violations = parse_output_options(options, &parsed);
    for (const auto &violation : option_violations) {
        request->violations.push_back({"/options" + violation.pointer, violation.message});
    }
    if (option_violations.empty()) {
        request->options = parsed;
    }
}

inline ParsedRequest bind_parsed_request(DecodeResult<Request> decoded,
                                         const std::vector<ParamSpec> &specs) {
    ParsedRequest request;
    request.req_id = std::move(decoded.value.req_id);
    request.status = decoded.status;
    append_decode_violations(decoded, &request);

    if (decoded.status == StatusCode::JSON_DECODE_ERROR) {
        return request;
    }

    for (auto &image : decoded.value.images) {
        byte_source item;
        item.origin = byte_source::origin_kind::base64_text;
        item.data = std::move(image);
        request.items.push_back(std::move(item));
    }

    if (decoded.value.has_params) {
        bind_params(decoded.value.params, specs, &request);
    }
    if (decoded.value.has_options) {
        bind_options(decoded.value.options, &request);
    }

    request.is_valid = request.violations.empty();
    if (decoded.status != StatusCode::JSON_DECODE_ERROR) {
        request.status = request.is_valid ? StatusCode::OK : StatusCode::INVALID_REQUEST_PARAMETER;
    }
    return request;
}

inline void apply_raw_headers(const std::string &params_header, const std::string &options_header,
                              const std::vector<ParamSpec> &specs, ParsedRequest *request) {
    if (!params_header.empty()) {
        rapidjson::Document header;
        header.Parse(params_header.data(), params_header.size());
        if (header.HasParseError() || !header.IsObject()) {
            request->violations.push_back({"/params", "X-Mortred-Params must be a compact JSON object"});
        } else {
            bind_params(header, specs, request);
        }
    }
    if (!options_header.empty()) {
        rapidjson::Document header;
        header.Parse(options_header.data(), options_header.size());
        if (header.HasParseError() || !header.IsObject()) {
            request->violations.push_back({"/options", "X-Mortred-Options must be a compact JSON object"});
        } else {
            bind_options(header, request);
        }
    }
}

} // namespace detail

/*** JSON encoding: decode the wire envelope, then bind ParamSpec / options. */
inline ParsedRequest parse_request_envelope(const std::string &body, const std::vector<ParamSpec> &specs) {
    return detail::bind_parsed_request(jinq::common::envelope::decode_request(body), specs);
}

/*** Raw-body encoding of the SAME envelope: Content-Type image-anything (or
 * application/octet-stream) carries exactly one image as the request body
 * (images[0]); params/options ride the X-Mortred-Params / X-Mortred-Options
 * headers as compact JSON; the optional trace id rides X-Request-ID. */
inline ParsedRequest parse_raw_request(const std::string &body, const std::string &req_id_header,
                                       const std::string &params_header, const std::string &options_header,
                                       const std::vector<ParamSpec> &specs) {
    ParsedRequest request;
    request.req_id = req_id_header;

    if (body.empty()) {
        request.status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        request.violations.push_back({"/body", "raw request body is empty (expected image bytes)"});
        return request;
    }

    byte_source item;
    item.origin = byte_source::origin_kind::raw_bytes;
    item.data = body;
    request.items.push_back(std::move(item));

    detail::apply_raw_headers(params_header, options_header, specs, &request);

    request.is_valid = request.violations.empty();
    request.status = request.is_valid ? StatusCode::OK : StatusCode::INVALID_REQUEST_PARAMETER;
    return request;
}

} // namespace server
} // namespace jinq

#endif // MORTRED_SERVER_PARSED_REQUEST_H
