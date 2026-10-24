/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: classification_topk_unittest.cc
* Date: 26-10-24
************************************************/

// The classification family's request-level top_k: convert one inline loop per
// model into one helper, so the boundaries are pinned here.

#include <vector>

#include <gtest/gtest.h>

#include "models/classification/top_k.h"

using mortred::models::classification::apply_top_k;

TEST(classification_top_k, keeps_the_highest_in_descending_order) {
    std::vector<float> scores{0.1f, 0.9f, 0.4f, 0.7f};
    apply_top_k(&scores, 2);
    ASSERT_EQ(scores.size(), 2u);
    EXPECT_FLOAT_EQ(scores[0], 0.9f);
    EXPECT_FLOAT_EQ(scores[1], 0.7f);
}

TEST(classification_top_k, keep_at_or_above_size_is_a_no_op) {
    std::vector<float> scores{0.3f, 0.6f};
    apply_top_k(&scores, 2);
    EXPECT_EQ(scores, (std::vector<float>{0.3f, 0.6f}));
    apply_top_k(&scores, 5);
    EXPECT_EQ(scores, (std::vector<float>{0.3f, 0.6f}));
}

TEST(classification_top_k, keep_zero_yields_empty_instead_of_reading_out_of_range) {
    // The HTTP ParamSpec only allows >= 1; a direct caller can still pass 0.
    std::vector<float> scores{0.3f, 0.6f};
    apply_top_k(&scores, 0);
    EXPECT_TRUE(scores.empty());
}

TEST(classification_top_k, empty_input_and_null_are_safe) {
    std::vector<float> scores;
    apply_top_k(&scores, 3);
    EXPECT_TRUE(scores.empty());
    apply_top_k(nullptr, 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
