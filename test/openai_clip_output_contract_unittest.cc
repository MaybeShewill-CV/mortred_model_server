#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "models/clip/openai_clip.h"
#include "models/io/clip.h"

using jinq::common::StatusCode;
using Model = jinq::models::clip::OpenAiClip<jinq::models::io_define::clip::clip_input, jinq::models::io_define::clip::clip_output>;

namespace {

template <typename T> class CallableClip : public T {
  public:
    using T::embedding_shape_status;
};

using TestModel = CallableClip<Model>;

} // namespace

TEST(OpenAiClipOutputContract, EmptyTextsStayModelEmptyOutput) {
    TestModel model;
    std::vector<float> scores;
    const auto status = model.texts2img({}, cv::Mat(), scores);
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_NE(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(OpenAiClipOutputContract, IncompatibleEmbeddingReturnsContractFailed) {
    const auto status = TestModel::embedding_shape_status({1.0f, 2.0f}, {1.0f});
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}
