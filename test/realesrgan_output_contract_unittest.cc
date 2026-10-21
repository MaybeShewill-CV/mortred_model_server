#include <gtest/gtest.h>

#include <ctime>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"
#include "models/enhancement/real_esrgan.h"
#include "models/io/enhancement.h"

using mortred::common::StatusCode;
using mortred::models::backend::InferenceContext;
using mortred::models::backend::NamedTensor;
using mortred::models::backend::Tensor;
using Output = mortred::models::io_define::enhancement::std_enhancement_output;

namespace {

template <typename MODEL> class CallableRealEsrGan : public MODEL {
  public:
    using MODEL::postprocess;
};

using TestModel = CallableRealEsrGan<mortred::models::enhancement::RealEsrGan<
    mortred::models::io_define::common_io::mat_input, Output>>;

class CaptureSink : public google::LogSink {
  public:
    void send(google::LogSeverity, const char *, const char *, int, const std::tm *, const char *message,
              size_t message_len) override {
        text.append(message, message_len);
        text.push_back('\n');
    }
    std::string text;
};

NamedTensor f32_tensor(const std::vector<int64_t> &shape) {
    NamedTensor tensor;
    tensor.name = "output";
    tensor.tensor = Tensor::make<float>(shape);
    return tensor;
}

} // namespace

TEST(RealEsrGanOutputContract, EmptyOutputsStayModelEmptyOutput) {
    TestModel model;
    Output out;
    CaptureSink sink;
    google::AddLogSink(&sink);
    const auto status = model.postprocess({}, InferenceContext{}, out);
    google::RemoveLogSink(&sink);
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(sink.text.find("real esrgan output tensor is empty"), std::string::npos) << sink.text;
}

TEST(RealEsrGanOutputContract, WrongChannelReturnsOutputContractFailed) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({f32_tensor({1, 4, 2, 2})}, InferenceContext{}, out);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(RealEsrGanOutputContract, WrongRankReturnsOutputContractFailed) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({f32_tensor({8, 8})}, InferenceContext{}, out);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(RealEsrGanOutputContract, ShortBufferReturnsOutputContractFailed) {
    TestModel model;
    Output out;
    auto tensor = f32_tensor({1, 3, 2, 2});
    tensor.tensor.buffer.resize(4);
    const auto status = model.postprocess({tensor}, InferenceContext{}, out);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(RealEsrGanOutputContract, ValidNchwReturnsOk) {
    TestModel model;
    Output out;
    const auto status = model.postprocess({f32_tensor({1, 3, 2, 2})}, InferenceContext{}, out);
    EXPECT_EQ(status, StatusCode::OK);
    ASSERT_FALSE(out.enhancement_result.empty());
    EXPECT_EQ(out.enhancement_result.size(), cv::Size(2, 2));
    EXPECT_EQ(out.enhancement_result.type(), CV_8UC3);
}
