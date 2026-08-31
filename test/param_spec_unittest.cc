/************************************************
 * Author: Codex
 * File: param_spec_unittest.cc
 * Date: 2026-08-31
 ************************************************/

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "models/backend/param_spec.h"

using jinq::models::backend::ParamSet;
using jinq::models::backend::ParamSpec;
using jinq::models::backend::ParamValue;
using jinq::models::backend::validate_params;

namespace {

std::vector<ParamSpec> sample_specs() {
    return {
        ParamSpec::f32("score_threshold").range(0.0, 1.0),
        ParamSpec::i32("top_k").range(1, 1000),
        ParamSpec::boolean("half_resolution"),
        ParamSpec::str("mask_encoding").values({"png", "jpeg"}),
        ParamSpec::i32("engine_threads").range(1, 32).config_only(),
    };
}

using Candidate = std::pair<std::string, ParamValue>;

} // namespace

TEST(param_spec, builders_capture_constraints) {
    const auto spec = ParamSpec::f32("score_threshold").range(0.0, 1.0).desc("detection confidence");
    EXPECT_EQ(spec.key, "score_threshold");
    EXPECT_EQ(spec.type, ParamSpec::Type::F32);
    EXPECT_TRUE(spec.has_range);
    EXPECT_DOUBLE_EQ(spec.range_min, 0.0);
    EXPECT_DOUBLE_EQ(spec.range_max, 1.0);
    EXPECT_EQ(spec.description, "detection confidence");
    EXPECT_TRUE(spec.request_overridable);
    EXPECT_STREQ(spec.type_name(), "f32");

    const auto config_only = ParamSpec::i32("threads").config_only();
    EXPECT_FALSE(config_only.request_overridable);

    const auto enumerated = ParamSpec::str("encoding").values({"png", "jpeg"});
    ASSERT_EQ(enumerated.enum_values.size(), 2u);
    EXPECT_EQ(enumerated.enum_values[1], "jpeg");
}

TEST(param_set, typed_roundtrip_and_fallbacks) {
    ParamSet params;
    EXPECT_TRUE(params.set_f32("score_threshold", 0.35f));
    EXPECT_TRUE(params.set_i32("top_k", 100));
    EXPECT_TRUE(params.set_bool("half", true));
    EXPECT_TRUE(params.set_str("encoding", "png"));

    EXPECT_FLOAT_EQ(params.get_f32("score_threshold", 0.5f), 0.35f);
    EXPECT_EQ(params.get_i32("top_k", 1), 100);
    EXPECT_TRUE(params.get_bool("half", false));
    EXPECT_EQ(params.get_str("encoding", "jpeg"), "png");

    // missing key / kind mismatch fall back to the config-derived default
    EXPECT_FLOAT_EQ(params.get_f32("missing", 0.25f), 0.25f);
    EXPECT_EQ(params.get_i32("encoding", 7), 7);

    EXPECT_TRUE(params.contains("top_k"));
    EXPECT_FALSE(params.contains("nope"));
    EXPECT_EQ(params.size(), 4u);
    EXPECT_EQ(params.keys().size(), 4u);
}

TEST(param_set, capacity_and_duplicate_writes_are_rejected) {
    ParamSet params;
    for (size_t idx = 0; idx < ParamSet::k_max_params; ++idx) {
        EXPECT_TRUE(params.set_i32("p" + std::to_string(idx), 1));
    }
    EXPECT_FALSE(params.set_i32("overflow", 1));
    EXPECT_EQ(params.size(), ParamSet::k_max_params);

    EXPECT_FALSE(params.set_i32("p0", 2));
    EXPECT_EQ(params.get_i32("p0", 0), 1);
}

TEST(validate_params, happy_path_canonicalises_all_types) {
    ParamSet params;
    const std::vector<Candidate> candidates = {
        {"score_threshold", ParamValue::of(0.35)},
        {"top_k", ParamValue::of(static_cast<int64_t>(42))},
        {"half_resolution", ParamValue::of(true)},
        {"mask_encoding", ParamValue::of(std::string("jpeg"))},
    };
    const auto violations = validate_params(sample_specs(), candidates, &params);
    EXPECT_TRUE(violations.empty());
    EXPECT_FLOAT_EQ(params.get_f32("score_threshold", 0.0f), 0.35f);
    EXPECT_EQ(params.get_i32("top_k", 0), 42);
    EXPECT_TRUE(params.get_bool("half_resolution", false));
    EXPECT_EQ(params.get_str("mask_encoding", ""), "jpeg");
}

TEST(validate_params, f32_accepts_integer_literals) {
    ParamSet params;
    const std::vector<Candidate> candidates = {{"score_threshold", ParamValue::of(static_cast<int64_t>(1))}};
    EXPECT_TRUE(validate_params(sample_specs(), candidates, &params).empty());
    EXPECT_FLOAT_EQ(params.get_f32("score_threshold", 0.0f), 1.0f);
}

TEST(validate_params, unknown_key_lists_allowed_keys) {
    ParamSet params;
    const std::vector<Candidate> candidates = {{"score_treshold", ParamValue::of(0.5)}}; // typo on purpose
    const auto violations = validate_params(sample_specs(), candidates, &params);
    ASSERT_EQ(violations.size(), 1u);
    EXPECT_EQ(violations[0].pointer, "/score_treshold");
    EXPECT_NE(violations[0].message.find("unknown parameter"), std::string::npos);
    EXPECT_NE(violations[0].message.find("score_threshold"), std::string::npos);
    EXPECT_TRUE(params.empty());
}

TEST(validate_params, config_only_key_is_rejected) {
    ParamSet params;
    const std::vector<Candidate> candidates = {{"engine_threads", ParamValue::of(static_cast<int64_t>(4))}};
    const auto violations = validate_params(sample_specs(), candidates, &params);
    ASSERT_EQ(violations.size(), 1u);
    EXPECT_EQ(violations[0].pointer, "/engine_threads");
    EXPECT_NE(violations[0].message.find("configuration-only"), std::string::npos);
    EXPECT_TRUE(params.empty());
}

TEST(validate_params, type_mismatches_are_rejected) {
    const std::vector<Candidate> bad = {
        {"score_threshold", ParamValue::of(true)},                                  // bool is not a number
        {"score_threshold", ParamValue::of(std::string("0.5"))},                    // text is not a number
        {"top_k", ParamValue::of(0.5)},                                             // fractional integer
        {"half_resolution", ParamValue::of(static_cast<int64_t>(1))},               // number is not a bool
        {"mask_encoding", ParamValue::of(static_cast<int64_t>(3))},                 // number is not a string
    };
    for (const auto &candidate : bad) {
        ParamSet params;
        const auto violations = validate_params(sample_specs(), {candidate}, &params);
        ASSERT_EQ(violations.size(), 1u) << "key=" << candidate.first;
        EXPECT_NE(violations[0].message.find("must be"), std::string::npos) << "key=" << candidate.first;
        EXPECT_TRUE(params.empty()) << "key=" << candidate.first;
    }
}

TEST(validate_params, range_is_inclusive_and_rejects_out_of_bounds) {
    const std::vector<double> accepted = {0.0, 0.5, 1.0};
    for (const double value : accepted) {
        ParamSet params;
        EXPECT_TRUE(validate_params(sample_specs(), {{"score_threshold", ParamValue::of(value)}}, &params).empty())
            << "value=" << value;
    }
    const std::vector<double> rejected = {-0.01, 1.01};
    for (const double value : rejected) {
        ParamSet params;
        const auto violations = validate_params(sample_specs(), {{"score_threshold", ParamValue::of(value)}}, &params);
        ASSERT_EQ(violations.size(), 1u);
        EXPECT_NE(violations[0].message.find("must be in ["), std::string::npos);
    }

    ParamSet params;
    EXPECT_FALSE(validate_params(sample_specs(), {{"top_k", ParamValue::of(static_cast<int64_t>(0))}}, &params).empty());
}

TEST(validate_params, string_enum_and_empty_string) {
    ParamSet params;
    EXPECT_TRUE(validate_params(sample_specs(), {{"mask_encoding", ParamValue::of(std::string("png"))}}, &params).empty());

    const auto violations =
        validate_params(sample_specs(), {{"mask_encoding", ParamValue::of(std::string("webp"))}}, &params);
    ASSERT_EQ(violations.size(), 1u);
    EXPECT_NE(violations[0].message.find("must be one of: png, jpeg"), std::string::npos);

    const auto empty = validate_params(sample_specs(), {{"mask_encoding", ParamValue::of(std::string(""))}}, &params);
    ASSERT_EQ(empty.size(), 1u);
    EXPECT_NE(empty[0].message.find("non-empty"), std::string::npos);
}

TEST(validate_params, duplicate_keys_are_rejected) {
    ParamSet params;
    const std::vector<Candidate> candidates = {
        {"top_k", ParamValue::of(static_cast<int64_t>(10))},
        {"top_k", ParamValue::of(static_cast<int64_t>(20))},
    };
    const auto violations = validate_params(sample_specs(), candidates, &params);
    ASSERT_EQ(violations.size(), 1u);
    EXPECT_EQ(violations[0].pointer, "/top_k");
    EXPECT_NE(violations[0].message.find("duplicate"), std::string::npos);
    EXPECT_TRUE(params.empty());
}

TEST(validate_params, output_is_untouched_when_any_candidate_fails) {
    ParamSet params;
    params.set_f32("score_threshold", 0.11f);
    const std::vector<Candidate> candidates = {
        {"score_threshold", ParamValue::of(0.9)},
        {"unknown_key", ParamValue::of(static_cast<int64_t>(1))},
    };
    EXPECT_FALSE(validate_params(sample_specs(), candidates, &params).empty());
    EXPECT_FLOAT_EQ(params.get_f32("score_threshold", 0.0f), 0.11f);
    EXPECT_EQ(params.size(), 1u);
}

TEST(validate_params, empty_candidates_is_valid) {
    ParamSet params;
    EXPECT_TRUE(validate_params(sample_specs(), {}, &params).empty());
    EXPECT_TRUE(params.empty());
}

TEST(validate_params, candidate_count_over_capacity_is_rejected) {
    std::vector<ParamSpec> specs;
    std::vector<Candidate> candidates;
    for (size_t idx = 0; idx <= ParamSet::k_max_params; ++idx) {
        const std::string key = "p" + std::to_string(idx);
        specs.push_back(ParamSpec::i32(key).range(0, 100));
        candidates.emplace_back(key, ParamValue::of(static_cast<int64_t>(1)));
    }
    ParamSet params;
    const auto violations = validate_params(specs, candidates, &params);
    ASSERT_EQ(violations.size(), 1u);
    EXPECT_EQ(violations[0].pointer, "/");
    EXPECT_NE(violations[0].message.find("too many parameters"), std::string::npos);
    EXPECT_TRUE(params.empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
