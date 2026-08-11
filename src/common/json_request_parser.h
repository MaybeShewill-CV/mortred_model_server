/************************************************
 * Author: Codex
 * File: json_request_parser.h
 * Date: 2026-08-11
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_JSON_REQUEST_PARSER_H
#define MORTRED_MODEL_SERVER_JSON_REQUEST_PARSER_H

#include <string>

#include "rapidjson/document.h"

#include "common/status_code.h"

namespace jinq {
namespace common {

/***
 * 通用图像服务请求：base64 图像内容 + 调用方追踪 id
 */
struct JsonRequest {
    std::string image_content;
    std::string task_id;
    bool is_valid = false;
    StatusCode parse_status = StatusCode::OK;
};

/***
 * 纯函数：解析图像服务请求体（JSON 格式）。
 * 任何解析失败都返回 is_valid=false 并携带精确的 parse_status，
 * 绝不抛出异常、绝不断言（外部输入路径不允许 CHECK/assert）。
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
        !doc["img_data"].IsString()) {
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
