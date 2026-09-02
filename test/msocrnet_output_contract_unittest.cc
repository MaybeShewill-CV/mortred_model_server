#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"
#include "models/scene_segmentation/msocrnet.h"

using jinq::common::StatusCode;
using jinq::models::backend::InferenceContext;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using SegOutput = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

namespace {

template <typename MODEL> class CallablePostprocess : public MODEL {
  public:
    using MODEL::postprocess;
};

using TestModel = CallablePostprocess<jinq::models::scene_segmentation::MsOcrNet<
    jinq::models::io_define::common_io::mat_input, SegOutput>>;

NamedTensor i32_mask(const std::vector<int64_t> &shape, const std::vector<int32_t> &values) {
    NamedTensor tensor;
    tensor.name = "argmax";
    tensor.tensor = Tensor::make<int32_t>(shape);
    std::memcpy(tensor.tensor.buffer.data(), values.data(), values.size() * sizeof(int32_t));
    return tensor;
}

NamedTensor i64_mask(const std::vector<int64_t> &shape, const std::vector<int64_t> &values) {
    NamedTensor tensor;
    tensor.name = "argmax";
    tensor.tensor = Tensor::make<int64_t>(shape);
    std::memcpy(tensor.tensor.buffer.data(), values.data(), values.size() * sizeof(int64_t));
    return tensor;
}

InferenceContext request_ctx(const cv::Size &network, const cv::Size &source) {
    InferenceContext ctx;
    ctx.network_size = network;
    ctx.source_size = source;
    return ctx;
}

} // namespace

TEST(MsOcrNetOutputContract, I32MaskFillsLinearlyIntoNetworkSizedResult) {
    TestModel model;
    // non-square network: the linear fill must land on the exact (row, col)
    // cells - a single-index Mat::at(i) would be read as (i, 0) and walk off
    // the heap for every i >= height
    const auto mask = i32_mask({1, 2, 3}, {0, 1, 2, 3, 4, 5});
    SegOutput result;
    const auto request = request_ctx(cv::Size(3, 2), cv::Size(3, 2));
    ASSERT_EQ(model.postprocess({mask}, request, result), StatusCode::OK);
    ASSERT_EQ(result.segmentation_result.size(), cv::Size(3, 2));
    // row-major expectation: value at linear index k lands on (k / 3, k % 3)
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 0), 0);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 1), 1);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 2), 2);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 0), 3);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 1), 4);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 2), 5);
}

TEST(MsOcrNetOutputContract, I64MaskConvertsAndFillsLinearly) {
    TestModel model;
    const auto mask = i64_mask({1, 3, 2}, {10, 11, 12, 13, 14, 15});
    SegOutput result;
    const auto request = request_ctx(cv::Size(2, 3), cv::Size(2, 3));
    ASSERT_EQ(model.postprocess({mask}, request, result), StatusCode::OK);
    ASSERT_EQ(result.segmentation_result.size(), cv::Size(2, 3));
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 0), 10);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 1), 11);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 0), 12);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 1), 13);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(2, 0), 14);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(2, 1), 15);
}

TEST(MsOcrNetOutputContract, MnnSqueezedLayoutFillsLinearly) {
    TestModel model;
    // mnn exports the argmax mask as [1,H,W,1]
    const auto mask = i32_mask({1, 2, 2, 1}, {7, 8, 9, 10});
    SegOutput result;
    const auto request = request_ctx(cv::Size(2, 2), cv::Size(2, 2));
    ASSERT_EQ(model.postprocess({mask}, request, result), StatusCode::OK);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 0), 7);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 1), 8);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 0), 9);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 1), 10);
}

TEST(MsOcrNetOutputContract, ResizesMaskToRequestSourceSize) {
    TestModel model;
    const auto mask = i32_mask({1, 2, 2}, {1, 2, 3, 4});
    SegOutput result;
    // INTER_NEAREST 2x upscale: dst(x, y) = src(x / 2, y / 2)
    const auto request = request_ctx(cv::Size(2, 2), cv::Size(4, 4));
    ASSERT_EQ(model.postprocess({mask}, request, result), StatusCode::OK);
    ASSERT_EQ(result.segmentation_result.size(), cv::Size(4, 4));
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 0), 1);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(1, 2), 2);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(2, 0), 3);
    EXPECT_EQ(result.segmentation_result.at<int32_t>(3, 3), 4);
}

TEST(MsOcrNetOutputContract, RejectsMalformedOutputs) {
    TestModel model;
    SegOutput result;
    const auto request = request_ctx(cv::Size(2, 2), cv::Size(2, 2));

    // empty output list
    EXPECT_EQ(model.postprocess({}, request, result), StatusCode::MODEL_EMPTY_OUTPUT);

    // non-integer dtype
    NamedTensor f32_tensor;
    f32_tensor.name = "argmax";
    f32_tensor.tensor = Tensor::make<float>({1, 2, 2});
    EXPECT_EQ(model.postprocess({f32_tensor}, request, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // unexpected rank (neither [1,H,W] nor [1,H,W,1])
    auto rank2 = i32_mask({2, 2}, {0, 1, 2, 3});
    EXPECT_EQ(model.postprocess({rank2}, request, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // buffer shorter than the declared shape
    auto short_buffer = i32_mask({1, 2, 2}, {0, 1, 2});
    EXPECT_EQ(model.postprocess({short_buffer}, request, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // invalid request geometry (no source size)
    EXPECT_EQ(model.postprocess({i32_mask({1, 2, 2}, {0, 1, 2, 3})}, InferenceContext{}, result),
              StatusCode::MODEL_EMPTY_INPUT_IMAGE);
}
