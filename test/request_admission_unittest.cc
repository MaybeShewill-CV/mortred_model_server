/************************************************
 * Author: Codex
 * File: request_admission_unittest.cc
 * Date: 2026-09-09
 ************************************************/

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "server/request_admission.h"

using mortred::server::ContentTypeKind;
using mortred::server::content_type_kind;
using mortred::server::body_exceeds_limit;
using mortred::server::declared_body_exceeds;
using mortred::server::item_count_exceeds;
using mortred::server::normalized_media_type;
using mortred::server::queue_would_overflow;
using mortred::server::try_reserve_queue;

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


TEST(request_admission, body_exceeds_limit_empty_cl_uses_actual_size) {
    // Empty Content-Length must not bypass: actual bytes decide.
    EXPECT_FALSE(body_exceeds_limit("", 1024, 1));
    EXPECT_TRUE(body_exceeds_limit("", 1048577, 1));
    EXPECT_TRUE(body_exceeds_limit("1048577", 0, 1));
    EXPECT_FALSE(body_exceeds_limit("1024", 1024, 1));
    EXPECT_TRUE(body_exceeds_limit("not-a-number", 2 * 1024 * 1024, 1));
}

TEST(request_admission, reservation_boundaries) {
    std::atomic<size_t> waiting{0};
    EXPECT_TRUE(try_reserve_queue(waiting, 2, 2));
    EXPECT_EQ(waiting.load(), 2u);
    EXPECT_FALSE(try_reserve_queue(waiting, 1, 2));
    EXPECT_FALSE(try_reserve_queue(waiting, 2, 2));
    EXPECT_EQ(waiting.load(), 2u) << "a rejected reservation must not change the count";

    waiting.fetch_sub(2);
    EXPECT_TRUE(try_reserve_queue(waiting, 1, 2));
    EXPECT_EQ(waiting.load(), 1u);
}

TEST(request_admission, reservation_uncapped_when_limit_is_non_positive) {
    std::atomic<size_t> waiting{0};
    EXPECT_TRUE(try_reserve_queue(waiting, 4, 0));
    EXPECT_TRUE(try_reserve_queue(waiting, 4, -1));
    EXPECT_EQ(waiting.load(), 8u);
}

TEST(request_admission, concurrent_reservations_never_exceed_the_cap) {
    // P1-5: the old load() check plus a later += let every handler thread pass
    // the cap; with one atomic reservation exactly `cap` slots are ever handed
    // out, and the published count equals the accepted work.
    constexpr size_t kCap = 8;
    std::atomic<size_t> waiting{0};
    std::atomic<size_t> accepted{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (int attempt = 0; attempt < 500; ++attempt) {
                if (try_reserve_queue(waiting, 1, static_cast<int>(kCap))) {
                    accepted.fetch_add(1);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(accepted.load(), kCap) << "exactly cap reservations may succeed";
    EXPECT_EQ(waiting.load(), kCap) << "the published count must match accepted work";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
