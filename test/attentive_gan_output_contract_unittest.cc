#include <gtest/gtest.h>

#include <vector>

#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"
#include "models/enhancement/attentive_gan_derain_net.h"
#include "models/io/enhancement.h"

using mortred::common::StatusCode;
using mortred::models::backend::InferenceContext;
using mortred::models::backend::NamedTensor;
using mortred::models::backend::Tensor;
using Output = mortred::models::io_define::enhancement::std_enhancement_output;

namespace {

template <typename MODEL> class CallableAttentiveGan : public MODEL {
  public:
    using MODEL::postprocess;
};

using TestModel = CallableAttentiveGan<mortred::models::enhancement::AttentiveGanDerain<
    mortred::models::io_define::common_io::mat_input, Output>>;

InferenceContext square_context() {
    InferenceContext ctx;
    ctx.source_size = cv::Size(2, 2);
    ctx.network_size = cv::Size(2, 2);
    return ctx;
}

NamedTensor f32_tensor(const std::vector<int64_t> &shape) {
    NamedTensor tensor;
    tensor.name = "output";
    tensor.tensor = Tensor::make<float>(shape);
    return tensor;
}

} // namespace

TEST(AttentiveGanOutputContract, EmptyOutputsStayModelEmptyOutput) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({}, square_context(), out);
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(AttentiveGanOutputContract, ShapeOrVolumeMismatchReturnsOutputContractFailed) {
    TestModel model;
    Output out;
    auto tensor = f32_tensor({1, 3, 2, 2});
    tensor.tensor.buffer.resize(4);
    const auto status = model.postprocess({tensor}, square_context(), out);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(AttentiveGanOutputContract, ValidNchwReturnsOk) {
    TestModel model;
    Output out;
    auto tensor = f32_tensor({1, 3, 2, 2});
    auto *data = tensor.tensor.data<float>();
    const auto count = tensor.tensor.element_count();
    for (int64_t idx = 0; idx < count; ++idx) {
        data[idx] = (idx % 2 == 0) ? 0.0f : 1.0f;
    }
    const auto status = model.postprocess({tensor}, square_context(), out);
    EXPECT_EQ(status, StatusCode::OK);
    ASSERT_FALSE(out.enhancement_result.empty());
    EXPECT_EQ(out.enhancement_result.size(), cv::Size(2, 2));
}
