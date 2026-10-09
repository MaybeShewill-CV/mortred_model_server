#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "models/backend/tensor.h"
#include "models/clip/openai_clip.h"
#include "models/io/clip.h"

using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using Model = jinq::models::clip::OpenAiClip<jinq::models::io_define::clip::clip_input, jinq::models::io_define::clip::clip_output>;

namespace {

template <typename T> class CallableClip : public T {
  public:
    using T::classify_embedding_output;
    using T::embedding_shape_status;
};

using TestModel = CallableClip<Model>;

NamedTensor output_tensor(Tensor tensor) {
    NamedTensor named;
    named.name = "output";
    named.tensor = std::move(tensor);
    return named;
}

Tensor finite_embedding() {
    auto tensor = Tensor::make<float>({1, 4});
    tensor.data<float>()[0] = 2.0f;
    return tensor;
}

void expect_embedding_rejected(StatusCode status, const std::vector<float> &embedding) {
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_EQ(embedding.size(), 0u);
}

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

TEST(OpenAiClipOutputContract, EmbeddingFiniteAccepted) {
    TestModel model;
    std::vector<float> embedding;
    const auto status = model.classify_embedding_output({output_tensor(finite_embedding())}, embedding);
    EXPECT_EQ(status, StatusCode::OK);
    ASSERT_EQ(embedding.size(), 4u);
    EXPECT_FLOAT_EQ(embedding[0], 2.0f);
    EXPECT_FLOAT_EQ(embedding[1], 0.0f);
    EXPECT_FLOAT_EQ(embedding[2], 0.0f);
    EXPECT_FLOAT_EQ(embedding[3], 0.0f);
}

TEST(OpenAiClipOutputContract, EmbeddingWrongDtype) {
    TestModel model;
    std::vector<float> embedding{1.0f};
    const auto status = model.classify_embedding_output({output_tensor(Tensor::make<int32_t>({1, 4}))}, embedding);
    expect_embedding_rejected(status, embedding);
}

TEST(OpenAiClipOutputContract, EmbeddingNan) {
    TestModel model;
    std::vector<float> embedding{1.0f};
    auto tensor = finite_embedding();
    tensor.data<float>()[1] = std::numeric_limits<float>::quiet_NaN();
    const auto status = model.classify_embedding_output({output_tensor(std::move(tensor))}, embedding);
    expect_embedding_rejected(status, embedding);
}

TEST(OpenAiClipOutputContract, EmbeddingPositiveInf) {
    TestModel model;
    std::vector<float> embedding{1.0f};
    auto tensor = finite_embedding();
    tensor.data<float>()[1] = std::numeric_limits<float>::infinity();
    const auto status = model.classify_embedding_output({output_tensor(std::move(tensor))}, embedding);
    expect_embedding_rejected(status, embedding);
}
