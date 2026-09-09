/************************************************
 * Author: Codex
 * File: request_admission_unittest.cc
 * Date: 2026-09-09
 ************************************************/

#include <gtest/gtest.h>

#include "server/request_admission.h"

using jinq::server::ContentTypeKind;
using jinq::server::content_type_kind;
using jinq::server::declared_body_exceeds;
using jinq::server::item_count_exceeds;
using jinq::server::normalized_media_type;
using jinq::server::queue_would_overflow;

TEST(request_admission, normalized_media_type_strips_params_and_case) {
    EXPECT_EQ(normalized_media_type("Application/JSON; charset=utf-8"), "application/json");
    EXPECT_EQ(normalized_media_type("  image/jpeg  "), "image/jpeg");
    EXPECT_EQ(normalized_media_type("application/octet-stream"), "application/octet-stream");
}

TEST(request_admission, content_type_kind_classifies_json_raw_and_other) {
    EXPECT_EQ(content_type_kind("application/json; charset=utf-8"), ContentTypeKind::Json);
    EXPECT_EQ(content_type_kind("IMAGE/PNG"), ContentTypeKind::RawBody);
    EXPECT_EQ(content_type_kind("application/octet-stream"), ContentTypeKind::RawBody);
    EXPECT_EQ(content_type_kind("text/plain"), ContentTypeKind::Unsupported);
    EXPECT_EQ(content_type_kind(""), ContentTypeKind::Unsupported);
}

TEST(request_admission, declared_body_exceeds_limit) {
    EXPECT_FALSE(declared_body_exceeds("", 1));
    EXPECT_FALSE(declared_body_exceeds("not-a-number", 1));
    EXPECT_FALSE(declared_body_exceeds("1024", 1));
    EXPECT_TRUE(declared_body_exceeds("1048577", 1));
}

TEST(request_admission, item_count_and_queue_predicates) {
    EXPECT_FALSE(item_count_exceeds(16, 16));
    EXPECT_TRUE(item_count_exceeds(17, 16));
    EXPECT_FALSE(queue_would_overflow(10, 2, 0));
    EXPECT_FALSE(queue_would_overflow(10, 2, -1));
    EXPECT_FALSE(queue_would_overflow(8, 2, 10));
    EXPECT_TRUE(queue_would_overflow(9, 2, 10));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
