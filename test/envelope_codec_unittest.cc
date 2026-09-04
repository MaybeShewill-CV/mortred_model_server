/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: envelope_codec_unittest.cc
 * Date: 26-9-4
 ************************************************/

// Wire-format roundtrips for the unified envelopes. Field names and
// encode/decode live only in common/request_envelope.h and
// common/response_envelope.h.

#include <string>

#include <gtest/gtest.h>

#include "common/request_envelope.h"
#include "common/response_envelope.h"
#include "common/status_code.h"

using jinq::common::StatusCode;
using jinq::common::UnifiedResponse;
using jinq::common::envelope::Request;
using jinq::common::envelope::decode_request;
using jinq::common::envelope::decode_response;
using jinq::common::envelope::encode;
using jinq::common::envelope::k_img_data_migration;

TEST(envelope_codec, request_roundtrip_preserves_fields) {
    Request request;
    request.req_id = "t1";
    request.images = {"aGk=", "ieU="};
    request.has_params = true;
    request.params.Parse(R"({"top_k":3})");
    request.has_options = true;
    request.options.Parse(R"({"include_image":false})");

    const auto decoded = decode_request(encode(request));
    ASSERT_TRUE(decoded.ok);
    EXPECT_EQ(decoded.value.req_id, "t1");
    ASSERT_EQ(decoded.value.images.size(), 2u);
    EXPECT_EQ(decoded.value.images[1], "ieU=");
    ASSERT_TRUE(decoded.value.has_params);
    EXPECT_EQ(decoded.value.params["top_k"].GetInt(), 3);
    ASSERT_TRUE(decoded.value.has_options);
    EXPECT_FALSE(decoded.value.options["include_image"].GetBool());
}

TEST(envelope_codec, request_img_data_never_succeeds) {
    const auto decoded = decode_request(R"({"img_data":"aGk=","images":["aGk="]})");
    EXPECT_FALSE(decoded.ok);
    EXPECT_EQ(decoded.status, StatusCode::INVALID_REQUEST_PARAMETER);
    ASSERT_EQ(decoded.violations.size(), 1u);
    EXPECT_EQ(decoded.violations[0].pointer, "/img_data");
    EXPECT_STREQ(decoded.violations[0].message.c_str(), k_img_data_migration);
}

TEST(envelope_codec, request_unknown_key_is_rejected) {
    const auto decoded = decode_request(R"({"images":["aGk="],"foo":1})");
    EXPECT_FALSE(decoded.ok);
    ASSERT_FALSE(decoded.violations.empty());
    EXPECT_EQ(decoded.violations[0].pointer, "/foo");
}

TEST(envelope_codec, response_roundtrip_and_unknown_keys_ignored) {
    UnifiedResponse resp;
    resp.task_id = "trace-1";
    resp.status = 0;
    resp.status_str = "OK";
    resp.model_name = "yolov8";
    resp.model_version = "sha256:abc";
    resp.server_time_ms = 41.25;
    resp.partial = false;

    jinq::common::ResponseItem ok_item;
    ok_item.status = 0;
    ok_item.data.SetObject();
    ok_item.data.AddMember("image", "abc", ok_item.data.GetAllocator());
    resp.results.push_back(std::move(ok_item));

    const std::string body = encode(resp);
    const auto decoded = decode_response(body);
    ASSERT_TRUE(decoded.ok);
    EXPECT_EQ(decoded.value.task_id, "trace-1");
    EXPECT_EQ(decoded.value.model_name, "yolov8");
    ASSERT_EQ(decoded.value.results.size(), 1u);
    EXPECT_STREQ(decoded.value.results[0].data["image"].GetString(), "abc");

    const auto with_extra = decode_response(R"({"status":0,"results":[{"status":0,"data":{"image":"x"}}],"future":true})");
    ASSERT_TRUE(with_extra.ok);
    ASSERT_EQ(with_extra.value.results.size(), 1u);
    EXPECT_STREQ(with_extra.value.results[0].data["image"].GetString(), "x");
}
