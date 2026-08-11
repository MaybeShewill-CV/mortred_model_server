/************************************************
 * Author: Codex
 * File: json_request_parser_unittest.cc
 * Date: 2026-08-11
 ************************************************/

#include <string>

#include <gtest/gtest.h>

#include "common/json_request_parser.h"

using jinq::common::JsonRequest;
using jinq::common::StatusCode;
using jinq::common::parse_json_request;

TEST(json_request_parser, parse_valid_request) {
    auto req = parse_json_request(R"({"img_data":"AAAA","req_id":"abc123"})");
    ASSERT_TRUE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::OK);
    EXPECT_EQ(req.image_content, "AAAA");
    EXPECT_EQ(req.task_id, "abc123");
}

TEST(json_request_parser, missing_req_id_is_still_valid) {
    auto req = parse_json_request(R"({"img_data":"AAAA"})");
    ASSERT_TRUE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::OK);
    EXPECT_EQ(req.image_content, "AAAA");
    EXPECT_TRUE(req.task_id.empty());
}

TEST(json_request_parser, top_level_array_is_rejected) {
    auto req = parse_json_request("[1,2,3]");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::JSON_DECODE_ERROR);
}

TEST(json_request_parser, top_level_scalar_is_rejected) {
    EXPECT_EQ(parse_json_request("\"abc\"").parse_status,
              StatusCode::JSON_DECODE_ERROR);
    EXPECT_EQ(parse_json_request("123").parse_status,
              StatusCode::JSON_DECODE_ERROR);
    EXPECT_EQ(parse_json_request("true").parse_status,
              StatusCode::JSON_DECODE_ERROR);
    EXPECT_EQ(parse_json_request("null").parse_status,
              StatusCode::JSON_DECODE_ERROR);
}

TEST(json_request_parser, malformed_json_is_rejected) {
    auto req = parse_json_request("{not valid json");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::JSON_DECODE_ERROR);
}

TEST(json_request_parser, empty_body_is_rejected) {
    auto req = parse_json_request("");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::JSON_DECODE_ERROR);
}

TEST(json_request_parser, empty_object_is_rejected) {
    auto req = parse_json_request("{}");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
}

TEST(json_request_parser, missing_img_data_is_rejected) {
    auto req = parse_json_request(R"({"req_id":"abc"})");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
}

TEST(json_request_parser, non_string_img_data_is_rejected) {
    auto req = parse_json_request(R"({"img_data":123})");
    EXPECT_FALSE(req.is_valid);
    EXPECT_EQ(req.parse_status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
