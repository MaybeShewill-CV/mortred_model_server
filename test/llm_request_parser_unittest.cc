/************************************************
 * Author: Codex
 * File: llm_request_parser_unittest.cc
 * Date: 2026-08-12
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/llm_request_parser.h"

using jinq::common::LlmChatRequest;
using jinq::common::StatusCode;
using jinq::common::parse_llm_chat_request;

TEST(llm_request_parser, valid_text_messages) {
    auto req = parse_llm_chat_request(
        R"({"task_id":"t1","data":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]})");
    ASSERT_TRUE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::OK);
    EXPECT_EQ(req.task_id, "t1");
    ASSERT_EQ(req.messages.size(), 2u);
    EXPECT_EQ(req.messages[0].first, "user");
    EXPECT_EQ(req.messages[0].second, "hello");
    EXPECT_EQ(req.messages[1].second, "hi");
}

TEST(llm_request_parser, valid_multimodal_array_content) {
    auto req = parse_llm_chat_request(
        R"({"data":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"http://x/a.jpg"}}]}]})");
    ASSERT_TRUE(req.is_valid);
    ASSERT_EQ(req.messages.size(), 1u);
    // 数组形式的 content 被序列化为 JSON 字符串，供 qwen2-vl 使用
    EXPECT_NE(req.messages[0].second.find("\"type\":\"image_url\""), std::string::npos);
}

TEST(llm_request_parser, empty_data_is_valid) {
    auto req = parse_llm_chat_request(R"({"data":[]})");
    EXPECT_TRUE(req.is_valid);
    EXPECT_TRUE(req.messages.empty());
}

TEST(llm_request_parser, task_id_must_be_string) {
    auto req = parse_llm_chat_request(R"({"task_id":123,"data":[]})");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::JSON_DECODE_ERROR);
}

TEST(llm_request_parser, top_level_non_object_rejected) {
    EXPECT_EQ(parse_llm_chat_request("[1,2]").parse_status, StatusCode::JSON_DECODE_ERROR);
    EXPECT_EQ(parse_llm_chat_request("not json").parse_status, StatusCode::JSON_DECODE_ERROR);
}

TEST(llm_request_parser, data_must_be_array) {
    EXPECT_EQ(parse_llm_chat_request(R"({"data":123})").parse_status, StatusCode::JSON_DECODE_ERROR);
    EXPECT_EQ(parse_llm_chat_request(R"({"data":"str"})").parse_status, StatusCode::JSON_DECODE_ERROR);
    EXPECT_EQ(parse_llm_chat_request(R"({"data":{"role":"user"}})").parse_status, StatusCode::JSON_DECODE_ERROR);
}

TEST(llm_request_parser, message_role_must_be_string) {
    auto req = parse_llm_chat_request(R"({"data":[{"role":1,"content":"x"}]})");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::JSON_DECODE_ERROR);
}

TEST(llm_request_parser, message_content_type_validated) {
    EXPECT_FALSE(parse_llm_chat_request(R"({"data":[{"role":"user","content":123}]})").is_valid);
    EXPECT_FALSE(parse_llm_chat_request(R"({"data":[{"role":"user"}]})").is_valid);
    EXPECT_FALSE(parse_llm_chat_request(R"({"data":["not-object"]})").is_valid);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
