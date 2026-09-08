#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "models/backend/nchw_host.h"

using jinq::models::backend::nchw_shape_from_nhwc4;
using jinq::models::backend::permute_nhwc_to_nchw_bytes;

TEST(NchwHost, RemapsRank4ShapeAndLeavesOthers) {
    EXPECT_EQ(nchw_shape_from_nhwc4({1, 8, 16, 3}), (std::vector<int64_t>{1, 3, 8, 16}));
    EXPECT_EQ(nchw_shape_from_nhwc4({1, 1000}), (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(nchw_shape_from_nhwc4({2, 3, 4}), (std::vector<int64_t>{2, 3, 4}));
}

TEST(NchwHost, PermutesFloatNhwcToNchw) {
    // NHWC [1,2,2,3]: pixel (h,w) channels packed last
    const float nhwc[] = {
        0, 1, 2,  // h0 w0
        3, 4, 5,  // h0 w1
        6, 7, 8,  // h1 w0
        9, 10, 11 // h1 w1
    };
    float nchw[12] = {};
    ASSERT_TRUE(permute_nhwc_to_nchw_bytes(reinterpret_cast<const uint8_t*>(nhwc),
                                           reinterpret_cast<uint8_t*>(nchw), 1, 2, 2, 3,
                                           sizeof(float)));
    // channel 0 plane: 0,3,6,9
    EXPECT_EQ(nchw[0], 0.0f);
    EXPECT_EQ(nchw[1], 3.0f);
    EXPECT_EQ(nchw[2], 6.0f);
    EXPECT_EQ(nchw[3], 9.0f);
    // channel 1 plane: 1,4,7,10
    EXPECT_EQ(nchw[4], 1.0f);
    EXPECT_EQ(nchw[5], 4.0f);
    EXPECT_EQ(nchw[6], 7.0f);
    EXPECT_EQ(nchw[7], 10.0f);
    // channel 2 plane: 2,5,8,11
    EXPECT_EQ(nchw[8], 2.0f);
    EXPECT_EQ(nchw[9], 5.0f);
    EXPECT_EQ(nchw[10], 8.0f);
    EXPECT_EQ(nchw[11], 11.0f);
}

TEST(NchwHost, RejectsInPlaceOrEmptyDims) {
    float buf[4] = {};
    auto* bytes = reinterpret_cast<uint8_t*>(buf);
    EXPECT_FALSE(permute_nhwc_to_nchw_bytes(bytes, bytes, 1, 1, 1, 1, sizeof(float)));
    EXPECT_FALSE(permute_nhwc_to_nchw_bytes(bytes, bytes + 1, 0, 1, 1, 1, sizeof(float)));
}
