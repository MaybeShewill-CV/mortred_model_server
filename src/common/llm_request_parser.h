/************************************************
 * Author: Codex
 * File: llm_request_parser.h
 * Date: 2026-08-12
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LLM_REQUEST_PARSER_H
#define MORTRED_MODEL_SERVER_LLM_REQUEST_PARSER_H

#include <string>
#include <utility>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/status_code.h"

namespace jinq {
namespace common {

/***
 * LLM 聊天请求：task_id（可选）+ 消息列表。
 * content 支持两种形态：纯文本字符串，或 qwen2-vl 多模态的数组（会被序列化为 JSON 字符串）。
 */
struct LlmChatRequest {
    std::string task_id;
    std::vector<std::pair<std::string, std::string> > messages; // <role, content>
    bool is_valid = false;
    StatusCode parse_status = StatusCode::OK;
};

/***
 * 纯函数：解析 LLM 聊天请求体（JSON）。
 * 逐字段做类型校验，任何非法输入都返回 is_valid=false + parse_status，
 * 绝不抛出异常、绝不断言、绝不访问错误类型的 JSON 值（外部输入路径不允许）。
 */
inline LlmChatRequest parse_llm_chat_request(const std::string& req_body) {
    LlmChatRequest req;

    rapidjson::Document doc;
    doc.Parse(req_body.data(), req_body.size());

    if (doc.HasParseError() || !doc.IsObject()) {
        req.parse_status = StatusCode::JSON_DECODE_ERROR;
        return req;
    }

    if (doc.HasMember("task_id")) {
        if (!doc["task_id"].IsString()) {
            req.parse_status = StatusCode::JSON_DECODE_ERROR;
            return req;
        }
        req.task_id = doc["task_id"].GetString();
    }

    if (!doc.HasMember("data") || !doc["data"].IsArray()) {
        req.parse_status = StatusCode::JSON_DECODE_ERROR;
        return req;
    }

    for (const auto& msg : doc["data"].GetArray()) {
        if (!msg.IsObject() || !msg.HasMember("role") || !msg["role"].IsString() ||
            !msg.HasMember("content")) {
            req.parse_status = StatusCode::JSON_DECODE_ERROR;
            return req;
        }

        std::string role = msg["role"].GetString();
        std::string content;
        if (msg["content"].IsString()) {
            content = msg["content"].GetString();
        } else if (msg["content"].IsArray()) {
            // qwen2-vl 多模态格式：把 content 数组序列化为 JSON 字符串
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            msg["content"].Accept(writer);
            content = buffer.GetString();
        } else {
            req.parse_status = StatusCode::JSON_DECODE_ERROR;
            return req;
        }

        req.messages.emplace_back(role, content);
    }

    req.is_valid = true;
    req.parse_status = StatusCode::OK;
    return req;
}

} // namespace common
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_LLM_REQUEST_PARSER_H
