/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: json_request_parser.h
* Date: 26-8-11
************************************************/

#ifndef MORTRED_MODEL_SERVER_JSON_REQUEST_PARSER_H
#define MORTRED_MODEL_SERVER_JSON_REQUEST_PARSER_H

#include <string>

#include "rapidjson/document.h"

#include "common/status_code.h"

namespace jinq {
namespace common {

/***
 * Generic image service request: base64 image content + caller trace id
 */
struct JsonRequest {
    std::string image_content;
    std::string task_id;
    bool is_valid = false;
    StatusCode parse_status = StatusCode::OK;
};

/***
 * Pure function: parse an image service request body (JSON).
 * Any failure returns is_valid=false with a precise parse_status.
 * Never throws and never asserts (external input paths forbid CHECK/assert).
 */
inline JsonRequest parse_json_request(const std::string& req_body) {
    JsonRequest req;

    rapidjson::Document doc;
    doc.Parse(req_body.data(), req_body.size());

    if (doc.HasParseError() || !doc.IsObject()) {
        req.parse_status = StatusCode::JSON_DECODE_ERROR;
        return req;
    }

    if (doc.ObjectEmpty() || !doc.HasMember("img_data") ||
        !doc["img_data"].IsString() || doc["img_data"].GetStringLength() == 0) {
        req.parse_status = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        return req;
    }

    req.image_content = doc["img_data"].GetString();
    if (doc.HasMember("req_id") && doc["req_id"].IsString()) {
        req.task_id = doc["req_id"].GetString();
    }

    req.is_valid = true;
    req.parse_status = StatusCode::OK;
    return req;
}

} // namespace common
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_JSON_REQUEST_PARSER_H
