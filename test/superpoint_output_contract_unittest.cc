#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "models/backend/tensor.h"
#include "models/feature_point/superpoint.h"
#include "models/io/feature_point.h"

using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::feature_point::SuperPoint;
using FP = jinq::models::io_define::feature_point::fp;
using FPOutput = jinq::models::io_define::feature_point::std_feature_point_output;

namespace {

template <typename MODEL> class CallableSuperPoint : public MODEL {
  public:
    using MODEL::_m_cell_size;
    using MODEL::_m_input_size_host;
    using MODEL::decode_fp_descriptor;
};

using TestModel = CallableSuperPoint<SuperPoint<jinq::models::io_define::common_io::mat_input, FPOutput>>;

// [1, 256, grid_row, grid_col] descriptor tensor with a fully predictable
// pattern: value(channel, row, col) = channel * 100 + row * 10 + col
NamedTensor patterned_desc(int grid_row, int grid_col) {
    const int64_t channels = 256;
    NamedTensor tensor;
    tensor.name = "output_2";
    tensor.tensor = Tensor::make<float>({1, channels, grid_row, grid_col});
    auto *data = tensor.tensor.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t r = 0; r < grid_row; ++r) {
            for (int64_t col = 0; col < grid_col; ++col) {
                data[c * grid_row * grid_col + r * grid_col + col] =
                    static_cast<float>(c * 100 + r * 10 + col);
            }
        }
    }
    return tensor;
}

std::vector<float> grid_descriptor(int row, int col) {
    std::vector<float> expected(256);
    for (int ch = 0; ch < 256; ++ch) {
        expected[ch] = static_cast<float>(ch * 100 + row * 10 + col);
    }
    return expected;
}

} // namespace

TEST(SuperPointDescriptorSampling, EdgeKeypointsClampIntoTheGrid) {
    // regression for the heap out-of-bounds read: keypoints in the last cell
    // row/column have ceil(x/y) == grid count, and the unclamped bilinear
    // sample read one 256-dim cell past the descriptor map. After clamping
    // they must degrade to the nearest grid cell (finite, exact values).
    TestModel model;
    model._m_cell_size = 8;
    model._m_input_size_host = cv::Size(64, 64); // 8x8 descriptor grid

    std::vector<FP> points = {
        {cv::Point2f(63.0f, 63.0f), {}, 1.0f}, // bottom-right: x/y = 7.875 -> cell (7,7)
        {cv::Point2f(63.0f, 32.0f), {}, 1.0f}, // right edge:   x = 7.875   -> cell (4,7)
        {cv::Point2f(32.0f, 63.0f), {}, 1.0f}, // bottom edge:  y = 7.875   -> cell (7,4)
        {cv::Point2f(0.0f, 0.0f), {}, 1.0f},   // top-left corner           -> cell (0,0)
        {cv::Point2f(16.0f, 16.0f), {}, 1.0f}, // interior cell center      -> cell (2,2)
    };

    model.decode_fp_descriptor(patterned_desc(8, 8), points);

    ASSERT_EQ(points.size(), 5u);
    for (const auto &p : points) {
        ASSERT_EQ(p.descriptor.size(), 256u) << p.location;
        for (const float v : p.descriptor) {
            EXPECT_TRUE(std::isfinite(v)) << p.location;
        }
    }
    EXPECT_EQ(points[0].descriptor, grid_descriptor(7, 7)) << points[0].location;
    EXPECT_EQ(points[1].descriptor, grid_descriptor(4, 7)) << points[1].location;
    EXPECT_EQ(points[2].descriptor, grid_descriptor(7, 4)) << points[2].location;
    EXPECT_EQ(points[3].descriptor, grid_descriptor(0, 0)) << points[3].location;
    EXPECT_EQ(points[4].descriptor, grid_descriptor(2, 2)) << points[4].location;
}

TEST(SuperPointDescriptorSampling, InteriorKeypointsStillInterpolate) {
    // non-edge samples must keep the true bilinear interpolation: x = 2.5
    // (location 20) sits between cells 2 and 3, weight 0.5 each
    TestModel model;
    model._m_cell_size = 8;
    model._m_input_size_host = cv::Size(64, 64);

    std::vector<FP> points = {
        {cv::Point2f(20.0f, 12.0f), {}, 1.0f}, // x = 2.5, y = 1.5
    };
    model.decode_fp_descriptor(patterned_desc(8, 8), points);
    ASSERT_EQ(points.size(), 1u);
    ASSERT_EQ(points[0].descriptor.size(), 256u);

    // expected: bilinear of cells (1,2),(1,3),(2,2),(2,3) with weight 0.5/0.5
    // value(c, r, col) = c*100 + r*10 + col; interpolation is linear per
    // channel, so the weighted result is c*100 + 15 + 2.5 per channel
    for (int ch = 0; ch < 256; ++ch) {
        EXPECT_NEAR(points[0].descriptor[ch], ch * 100.0f + 17.5f, 1e-3f) << "channel " << ch;
    }
}
