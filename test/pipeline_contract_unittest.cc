/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: pipeline_contract_unittest.cc
 * Date: 26-9-4
 ************************************************/

// Contract tests for the management-plane rewrite of the unified envelope:
// infer/jobs/pipelines must strip control-only keys, reject img_data with the
// same 422 migration text as the data plane, and pull pipeline hops from
// results[].data rather than the removed {data: ...} response.

#include <string>

#include <gtest/gtest.h>

#include "control/management_envelope.h"
#include "common/request_envelope.h"

using mortred::control::apply_pipeline_step_input;
using mortred::control::copy_request_envelope;
using mortred::control::extract_prev_output_images;

namespace {

rapidjson::Document parse_object(const std::string &json) {
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    EXPECT_FALSE(doc.HasParseError());
    EXPECT_TRUE(doc.IsObject());
    return doc;
}

} // namespace

TEST(pipeline_contract, img_data_migration_text_matches_data_plane) {
    EXPECT_STREQ(mortred::control::k_img_data_migration, jinq::common::envelope::k_img_data_migration);
}

TEST(pipeline_contract, infer_proxy_strips_server_id_and_keeps_envelope) {
    const auto doc = parse_object(
        R"({"server_id":"YOLOV8","req_id":"t1","images":["aGk="],"params":{"top_k":1}})");
    const auto rewrite = copy_request_envelope(doc);
    ASSERT_TRUE(rewrite.ok);
    EXPECT_EQ(rewrite.http_status, 200);

    const auto forwarded = parse_object(rewrite.json);
    EXPECT_FALSE(forwarded.HasMember("server_id"));
    ASSERT_TRUE(forwarded.HasMember("req_id"));
    EXPECT_STREQ(forwarded["req_id"].GetString(), "t1");
    ASSERT_TRUE(forwarded.HasMember("images"));
    ASSERT_TRUE(forwarded["images"].IsArray());
    ASSERT_EQ(forwarded["images"].Size(), 1u);
    EXPECT_STREQ(forwarded["images"][0].GetString(), "aGk=");
    ASSERT_TRUE(forwarded.HasMember("params"));
    EXPECT_EQ(forwarded["params"]["top_k"].GetInt(), 1);
}

TEST(pipeline_contract, jobs_and_pipelines_drop_management_keys) {
    const auto doc = parse_object(
        R"({"model":"REALESRGAN","steps":[{"model":"YOLOV8"}],"images":["aGk="],"options":{"include_image":true}})");
    const auto rewrite = copy_request_envelope(doc);
    ASSERT_TRUE(rewrite.ok);
    const auto forwarded = parse_object(rewrite.json);
    EXPECT_FALSE(forwarded.HasMember("model"));
    EXPECT_FALSE(forwarded.HasMember("steps"));
    ASSERT_TRUE(forwarded.HasMember("images"));
    EXPECT_STREQ(forwarded["images"][0].GetString(), "aGk=");
    ASSERT_TRUE(forwarded.HasMember("options"));
    EXPECT_TRUE(forwarded["options"]["include_image"].GetBool());
}

TEST(pipeline_contract, img_data_is_422_even_alongside_images) {
    for (const char *body : {R"({"server_id":"YOLOV8","img_data":"aGk="})",
                             R"({"server_id":"YOLOV8","img_data":"aGk=","images":["aGk="]})"}) {
        const auto rewrite = copy_request_envelope(parse_object(body));
        EXPECT_FALSE(rewrite.ok);
        EXPECT_EQ(rewrite.http_status, 422);
        EXPECT_EQ(rewrite.pointer, "/img_data");
        EXPECT_NE(rewrite.message.find("img_data -> images[0]"), std::string::npos);
    }
}

TEST(pipeline_contract, missing_images_is_rejected) {
    const auto rewrite = copy_request_envelope(parse_object(R"({"server_id":"YOLOV8","req_id":"t"})"));
    EXPECT_FALSE(rewrite.ok);
    EXPECT_EQ(rewrite.http_status, 422);
    EXPECT_EQ(rewrite.pointer, "/images");
}

TEST(pipeline_contract, prev_output_reads_results_data_not_legacy_data) {
    const auto rewrite = extract_prev_output_images(
        R"({"status":0,"results":[{"status":0,"data":{"image":"abc","colorized_mask":"def"}}]})",
        "image");
    ASSERT_TRUE(rewrite.ok) << rewrite.message;
    const auto next = parse_object(rewrite.json);
    ASSERT_TRUE(next.HasMember("images"));
    ASSERT_EQ(next["images"].Size(), 1u);
    EXPECT_STREQ(next["images"][0].GetString(), "abc");
    EXPECT_FALSE(next.HasMember("img_data"));
}

TEST(pipeline_contract, prev_output_wraps_string_array_as_images) {
    const auto rewrite = extract_prev_output_images(
        R"({"results":[{"data":{"crops":["aa","bb"]}}]})", "crops");
    ASSERT_TRUE(rewrite.ok) << rewrite.message;
    const auto next = parse_object(rewrite.json);
    ASSERT_EQ(next["images"].Size(), 2u);
    EXPECT_STREQ(next["images"][0].GetString(), "aa");
    EXPECT_STREQ(next["images"][1].GetString(), "bb");
}

TEST(pipeline_contract, prev_output_rejects_legacy_data_object) {
    const auto rewrite =
        extract_prev_output_images(R"({"data":{"image":"abc"}})", "image");
    EXPECT_FALSE(rewrite.ok);
    EXPECT_NE(rewrite.message.find("results[].data"), std::string::npos);
}

TEST(pipeline_contract, prev_output_rejects_missing_field) {
    const auto rewrite = extract_prev_output_images(
        R"({"results":[{"data":{"image":"abc"}}]})", "colorized_mask");
    EXPECT_FALSE(rewrite.ok);
    EXPECT_NE(rewrite.message.find("colorized_mask"), std::string::npos);
}

TEST(pipeline_contract, step_input_passthrough_keeps_request_envelope) {
    const std::string body = R"({"images":["aGk="],"params":{"top_k":3}})";
    const auto rewrite = apply_pipeline_step_input(body, "images");
    ASSERT_TRUE(rewrite.ok);
    EXPECT_EQ(rewrite.json, body);
}

TEST(pipeline_contract, step_input_prev_output_rebuilds_images_envelope) {
    const auto rewrite = apply_pipeline_step_input(
        R"({"task_id":"t","results":[{"status":0,"data":{"image":"xyz"}}]})",
        "prev_output.image");
    ASSERT_TRUE(rewrite.ok) << rewrite.message;
    const auto next = parse_object(rewrite.json);
    ASSERT_EQ(next["images"].Size(), 1u);
    EXPECT_STREQ(next["images"][0].GetString(), "xyz");
    EXPECT_FALSE(next.HasMember("img_data"));
    EXPECT_FALSE(next.HasMember("results"));
}
