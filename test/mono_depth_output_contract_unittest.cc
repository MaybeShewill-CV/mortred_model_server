#include <gtest/gtest.h>

#include <ctime>
#include <string>
#include <vector>

#include "glog/logging.h"

// DepthAnything::postprocess is private. The signed design forbids editing
// src/models/mono_depth_estimation. Promote access only in this TU so the
// test subclass can call postprocess, matching enlightengan's protected hook.
#define private public
#include "models/mono_depth_estimation/depth_anything.h"
#undef private

#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"
#include "models/io/mono_depth_estimation.h"

using jinq::common::StatusCode;
using jinq::models::backend::InferenceContext;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using Output = jinq::models::io_define::mono_depth_estimation::std_mde_output;

namespace {

template <typename MODEL> class CallableDepth : public MODEL {
  public:
    using MODEL::postprocess;
};

using TestModel = CallableDepth<jinq::models::mono_depth_estimation::DepthAnything<
    jinq::models::io_define::common_io::mat_input, Output>>;

class CaptureSink : public google::LogSink {
  public:
    void send(google::LogSeverity, const char *, const char *, int, const std::tm *, const char *message,
              size_t message_len) override {
        text.append(message, message_len);
        text.push_back('\n');
    }
    std::string text;
};

InferenceContext square_context() {
    InferenceContext ctx;
    ctx.source_size = cv::Size(8, 8);
    ctx.network_size = cv::Size(8, 8);
    return ctx;
}

} // namespace

TEST(MonoDepthOutputContract, EmptyOutputsFailWithDepthAnythingText) {
    TestModel model;
    Output out;
    CaptureSink sink;
    google::AddLogSink(&sink);
    const auto status = model.postprocess({}, square_context(), out);
    google::RemoveLogSink(&sink);
    EXPECT_NE(status, StatusCode::OK);
    EXPECT_NE(sink.text.find("depth anything"), std::string::npos) << sink.text;
}

TEST(MonoDepthOutputContract, WrongRankFailsWithShapeOrDepthText) {
    TestModel model;
    Output out;
    NamedTensor tensor;
    tensor.name = "output";
    tensor.tensor = Tensor::make<float>({8, 8});
    CaptureSink sink;
    google::AddLogSink(&sink);
    const auto status = model.postprocess({tensor}, square_context(), out);
    google::RemoveLogSink(&sink);
    EXPECT_NE(status, StatusCode::OK);
    const bool named = sink.text.find("depth anything") != std::string::npos ||
                       sink.text.find("shape") != std::string::npos;
    EXPECT_TRUE(named) << sink.text;
}
