#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "models/backend/inference_context.h"
#include "models/backend/model_runtime.h"
#include "models/backend/tensor.h"
#include "models/io/scene_segmentation.h"
#include "models/scene_segmentation/hrnet_segmentation.h"

using mortred::common::StatusCode;
using mortred::models::backend::InferenceContext;
using mortred::models::backend::NamedTensor;
using mortred::models::backend::Tensor;
using Output = mortred::models::io_define::scene_segmentation::std_scene_segmentation_output;

namespace {

template <typename MODEL> class CallableHrnet : public MODEL {
  public:
    using MODEL::postprocess;
};

using TestModel = CallableHrnet<mortred::models::scene_segmentation::HRNetSegmentation<
    mortred::models::io_define::common_io::mat_input, Output>>;

InferenceContext map_context() {
    InferenceContext ctx;
    ctx.source_size = cv::Size(8, 8);
    ctx.network_size = cv::Size(8, 8);
    return ctx;
}

NamedTensor i32_mask(const std::vector<int64_t> &shape) {
    NamedTensor tensor;
    tensor.name = "output";
    tensor.tensor = Tensor::make<int32_t>(shape);
    return tensor;
}

} // namespace

TEST(HRNetOutputContract, EmptyOutputsStayModelEmptyOutput) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({}, map_context(), out);
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(HRNetOutputContract, MaskSmallerThanNetworkReturnsOutputContractFailed) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({i32_mask({2, 2})}, map_context(), out);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(HRNetOutputContract, WrongRankStaysOutputContractFailed) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({i32_mask({8})}, map_context(), out);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(HRNetOutputContract, ValidMaskReturnsOk) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({i32_mask({8, 8})}, map_context(), out);
    EXPECT_EQ(status, StatusCode::OK);
    ASSERT_FALSE(out.segmentation_result.empty());
    EXPECT_EQ(out.segmentation_result.size(), cv::Size(8, 8));
}
