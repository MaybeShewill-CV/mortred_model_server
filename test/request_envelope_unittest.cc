/************************************************
 * Author: Codex
 * File: request_envelope_unittest.cc
 * Date: 2026-08-31
 ************************************************/

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "models/backend/param_spec.h"
#include "server/request_envelope.h"

using jinq::models::backend::ParamSpec;
using jinq::models::io_define::common_io::byte_source;
using jinq::server::parse_raw_request;
using jinq::server::OutputOptions;
using jinq::server::parse_request_envelope;
using jinq::common::StatusCode;

namespace {

std::vector<ParamSpec> sample_specs() {
    return {
        ParamSpec::f32("score_threshold").range(0.0, 1.0),
        ParamSpec::i32("top_k").range(1, 1000),
        ParamSpec::boolean("half_resolution"),
        ParamSpec::str("mask_encoding").values({"png", "jpeg"}),
    };
}

} // namespace

TEST(request_envelope, valid_single_image_with_defaults) {
    const auto request = parse_request_envelope(R"({"images":["aGVsbG8="]})", sample_specs());
    EXPECT_TRUE(request.is_valid);
    EXPECT_EQ(request.status, StatusCode::OK);
    EXPECT_TRUE(request.violations.empty());
    EXPECT_TRUE(request.req_id.empty());
    ASSERT_EQ(request.items.size(), 1u);
    EXPECT_EQ(request.items[0].origin, byte_source::origin_kind::base64_text);
    EXPECT_EQ(request.items[0].data, "aGVsbG8=");
    EXPECT_EQ(request.params, nullptr);

    // default output options
    EXPECT_EQ(request.options.encoding, OutputOptions::ImageEncoding::PNG);
    EXPECT_TRUE(request.options.include_image);
    EXPECT_EQ(request.options.max_results, 0);
    EXPECT_FALSE(request.options.echo_params);
}

TEST(request_envelope, valid_multi_image_with_req_id_and_options) {
    const auto request = parse_request_envelope(
        R"({"req_id":"trace-42","images":["aGk=","ieU="],"params":{"score_threshold":0.4,"top_k":50},)"
        R"("options":{"encoding":"jpeg","include_image":false,"max_results":10,"echo_params":true}})",
        sample_specs());
    EXPECT_TRUE(request.is_valid);
    EXPECT_EQ(request.req_id, "trace-42");
    ASSERT_EQ(request.items.size(), 2u);
    EXPECT_EQ(request.items[1].data, "ieU=");
    ASSERT_NE(request.params, nullptr);
    EXPECT_FLOAT_EQ(request.params->get_f32("score_threshold", 0.0f), 0.4f);
    EXPECT_EQ(request.params->get_i32("top_k", 0), 50);
    EXPECT_EQ(request.options.encoding, OutputOptions::ImageEncoding::JPEG);
    EXPECT_FALSE(request.options.include_image);
    EXPECT_EQ(request.options.max_results, 10);
    EXPECT_TRUE(request.options.echo_params);
}

TEST(request_envelope, empty_params_object_is_distinct_from_absent) {
    const auto request = parse_request_envelope(R"({"images":["aGk="],"params":{}})", sample_specs());
    EXPECT_TRUE(request.is_valid);
    ASSERT_NE(request.params, nullptr);
    EXPECT_TRUE(request.params->empty());
}

TEST(request_envelope, malformed_or_non_object_bodies) {
    const std::vector<std::string> bad_bodies = {
        "",
        "{nope",
        "[1,2,3]",
        "\"a string\"",
        "42",
    };
    for (const auto &body : bad_bodies) {
        const auto request = parse_request_envelope(body, sample_specs());
        EXPECT_FALSE(request.is_valid);
        EXPECT_EQ(request.status, StatusCode::JSON_DECODE_ERROR);
        ASSERT_EQ(request.violations.size(), 1u);
        EXPECT_EQ(request.violations[0].pointer, "/");
    }
}

TEST(request_envelope, unknown_top_level_field_lists_allowed) {
    const auto request = parse_request_envelope(R"({"images":["aGk="],"foo":1})", sample_specs());
    EXPECT_FALSE(request.is_valid);
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/foo");
    EXPECT_NE(request.violations[0].message.find("unknown request field 'foo'"), std::string::npos);
    EXPECT_NE(request.violations[0].message.find("req_id, images, params, options"), std::string::npos);
}

TEST(request_envelope, img_data_never_succeeds_even_alongside_images) {
    auto request = parse_request_envelope(R"({"img_data":"aGVsbG8="})", sample_specs());
    EXPECT_FALSE(request.is_valid);
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/img_data");
    EXPECT_NE(request.violations[0].message.find("img_data -> images[0]"), std::string::npos);

    request = parse_request_envelope(R"({"img_data":"aGVsbG8=","images":["aGVsbG8="]})", sample_specs());
    EXPECT_FALSE(request.is_valid);
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/img_data");
    EXPECT_EQ(request.status, StatusCode::INVALID_REQUEST_PARAMETER);
}

TEST(request_envelope, images_shape_errors) {
    struct Case {
        std::string body;
        std::string pointer;
        std::string fragment;
    };
    const std::vector<Case> cases = {
        {R"({"req_id":"a"})", "/images", "missing"},
        {R"({"images":"aGk="})", "/images", "must be an array"},
        {R"({"images":[]})", "/images", "at least one"},
        {R"({"images":["ok",5]})", "/images/1", "must be a string"},
        {R"({"images":[""]})", "/images/0", "non-empty"},
    };
    for (const auto &item : cases) {
        const auto request = parse_request_envelope(item.body, sample_specs());
        EXPECT_FALSE(request.is_valid) << item.body;
        ASSERT_EQ(request.violations.size(), 1u) << item.body;
        EXPECT_EQ(request.violations[0].pointer, item.pointer) << item.body;
        EXPECT_NE(request.violations[0].message.find(item.fragment), std::string::npos) << item.body;
    }
}

TEST(request_envelope, param_violations_are_prefixed_and_list_allowed) {
    const auto request =
        parse_request_envelope(R"({"images":["aGk="],"params":{"score_treshold":0.5}})", sample_specs());
    EXPECT_FALSE(request.is_valid);
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/params/score_treshold");
    EXPECT_NE(request.violations[0].message.find("unknown parameter"), std::string::npos);
    EXPECT_NE(request.violations[0].message.find("score_threshold"), std::string::npos);
}

TEST(request_envelope, param_type_range_and_scalar_errors) {
    struct Case {
        std::string params_json;
        std::string pointer;
        std::string fragment;
    };
    const std::vector<Case> cases = {
        {R"("params":{"top_k":0.5})", "/params/top_k", "must be an integer"},
        {R"("params":{"score_threshold":2.0})", "/params/score_threshold", "must be in ["},
        {R"("params":{"score_threshold":true})", "/params/score_threshold", "must be a number"},
        {R"("params":{"mask_encoding":"webp"})", "/params/mask_encoding", "must be one of: png, jpeg"},
        {R"("params":{"top_k":null})", "/params/top_k", "must be a scalar"},
        {R"("params":{"top_k":[1]})", "/params/top_k", "must be a scalar"},
        {R"("params":{"top_k":{"a":1}})", "/params/top_k", "must be a scalar"},
    };
    for (const auto &item : cases) {
        const auto request = parse_request_envelope(R"({"images":["aGk="],)" + item.params_json + "}", sample_specs());
        EXPECT_FALSE(request.is_valid) << item.params_json;
        ASSERT_EQ(request.violations.size(), 1u) << item.params_json;
        EXPECT_EQ(request.violations[0].pointer, item.pointer) << item.params_json;
        EXPECT_NE(request.violations[0].message.find(item.fragment), std::string::npos) << item.params_json;
        EXPECT_EQ(request.params, nullptr) << item.params_json;
    }
}

TEST(request_envelope, param_count_over_capacity_reports_params_root) {
    std::string params = "\"params\":{";
    for (size_t idx = 0; idx <= jinq::models::backend::ParamSet::k_max_params; ++idx) {
        if (idx != 0) {
            params += ",";
        }
        params += "\"k" + std::to_string(idx) + "\":1";
    }
    params += "}";
    const auto request = parse_request_envelope(R"({"images":["aGk="],)" + params + "}", {});
    EXPECT_FALSE(request.is_valid);
    ASSERT_FALSE(request.violations.empty());
    // capacity fires before the (schema-less) unknown-key checks would list everything
    EXPECT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/params");
    EXPECT_NE(request.violations[0].message.find("too many parameters"), std::string::npos);
}

TEST(request_envelope, option_errors_are_prefixed) {
    struct Case {
        std::string options_json;
        std::string pointer;
        std::string fragment;
    };
    const std::vector<Case> cases = {
        {R"("options":{"foo":1})", "/options/foo", "unknown option"},
        {R"("options":{"encoding":"bmp"})", "/options/encoding", "must be one of: png, jpeg, webp"},
        {R"("options":{"encoding":3})", "/options/encoding", "must be a string"},
        {R"("options":{"include_image":"yes"})", "/options/include_image", "must be a boolean"},
        {R"("options":{"max_results":-1})", "/options/max_results", "non-negative"},
        {R"("options":{"max_results":1.5})", "/options/max_results", "non-negative"},
    };
    for (const auto &item : cases) {
        const auto request =
            parse_request_envelope(R"({"images":["aGk="],)" + item.options_json + "}", sample_specs());
        EXPECT_FALSE(request.is_valid) << item.options_json;
        ASSERT_EQ(request.violations.size(), 1u) << item.options_json;
        EXPECT_EQ(request.violations[0].pointer, item.pointer) << item.options_json;
        EXPECT_NE(request.violations[0].message.find(item.fragment), std::string::npos) << item.options_json;
    }
}

TEST(request_envelope, multiple_violations_accumulate) {
    const auto request = parse_request_envelope(R"({"foo":1,"images":[],"req_id":7})", sample_specs());
    EXPECT_FALSE(request.is_valid);
    EXPECT_EQ(request.violations.size(), 3u);
    // check order: legacy field -> unknown fields -> req_id -> images
    EXPECT_EQ(request.violations[0].pointer, "/foo");
    EXPECT_EQ(request.violations[1].pointer, "/req_id");
    EXPECT_EQ(request.violations[2].pointer, "/images");
}

TEST(request_envelope, options_partial_override_keeps_defaults) {
    const auto request = parse_request_envelope(R"({"images":["aGk="],"options":{"encoding":"webp"}})",
                                                 sample_specs());
    EXPECT_TRUE(request.is_valid);
    EXPECT_EQ(request.options.encoding, OutputOptions::ImageEncoding::WEBP);
    EXPECT_STREQ(request.options.encoding_extension(), ".webp");
    EXPECT_TRUE(request.options.include_image);
    EXPECT_EQ(request.options.max_results, 0);
}

TEST(request_envelope, raw_body_builds_single_raw_item) {
    const auto request = parse_raw_request("\x89PNG-data", "trace-7", "", "", sample_specs());
    EXPECT_TRUE(request.is_valid);
    EXPECT_EQ(request.status, StatusCode::OK);
    EXPECT_EQ(request.req_id, "trace-7");
    ASSERT_EQ(request.items.size(), 1u);
    EXPECT_EQ(request.items[0].origin, byte_source::origin_kind::raw_bytes);
    EXPECT_EQ(request.items[0].data, std::string("\x89PNG-data"));
    EXPECT_EQ(request.params, nullptr);
    EXPECT_EQ(request.options.encoding, OutputOptions::ImageEncoding::PNG);
}

TEST(request_envelope, raw_empty_body_rejects_with_pointer) {
    const auto request = parse_raw_request("", "", "", "", sample_specs());
    EXPECT_FALSE(request.is_valid);
    EXPECT_EQ(request.status, StatusCode::MODEL_EMPTY_INPUT_IMAGE);
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/body");
}

TEST(request_envelope, raw_params_header_uses_the_same_validator_and_pointers) {
    // valid override
    auto request = parse_raw_request("img", "t", R"({"score_threshold":0.4})", "", sample_specs());
    EXPECT_TRUE(request.is_valid);
    ASSERT_NE(request.params, nullptr);
    EXPECT_FLOAT_EQ(request.params->get_f32("score_threshold", 0.0f), 0.4f);

    // unknown key: same pointer namespace as the JSON encoding
    request = parse_raw_request("img", "t", R"({"nope":1})", "", sample_specs());
    EXPECT_FALSE(request.is_valid);
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/params/nope");
    EXPECT_NE(request.violations[0].message.find("unknown parameter"), std::string::npos);

    // range violation is byte-identical semantics to the JSON path
    const auto json_twin =
        parse_request_envelope(R"({"images":["aGk="],"params":{"score_threshold":2.0}})", sample_specs());
    request = parse_raw_request("img", "t", R"({"score_threshold":2.0})", "", sample_specs());
    ASSERT_EQ(json_twin.violations.size(), request.violations.size());
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(json_twin.violations[0].pointer, request.violations[0].pointer);
    EXPECT_EQ(json_twin.violations[0].message, request.violations[0].message);

    // malformed header value
    request = parse_raw_request("img", "t", "not-json", "", sample_specs());
    EXPECT_FALSE(request.is_valid);
    EXPECT_EQ(request.violations[0].pointer, "/params");
    EXPECT_NE(request.violations[0].message.find("X-Mortred-Params"), std::string::npos);
}

TEST(request_envelope, raw_options_header_parsers_like_the_json_path) {
    auto request = parse_raw_request("img", "", "", R"({"encoding":"jpeg","max_results":5})",
                                     sample_specs());
    EXPECT_TRUE(request.is_valid);
    EXPECT_EQ(request.options.encoding, OutputOptions::ImageEncoding::JPEG);
    EXPECT_EQ(request.options.max_results, 5);

    request = parse_raw_request("img", "", "", R"({"encoding":"bmp"})", sample_specs());
    EXPECT_FALSE(request.is_valid);
    ASSERT_EQ(request.violations.size(), 1u);
    EXPECT_EQ(request.violations[0].pointer, "/options/encoding");

    request = parse_raw_request("img", "", "", "nope", sample_specs());
    EXPECT_FALSE(request.is_valid);
    EXPECT_EQ(request.violations[0].pointer, "/options");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
