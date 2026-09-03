#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "models/backend/inference_context.h"
#include "models/backend/param_spec.h"
#include "models/backend/tensor.h"
#include "models/feature_embedding/dinov2.h"
#include "models/io/feature_embedding.h"
#include "models/model_io_define.h"

using jinq::common::StatusCode;
using jinq::models::backend::InferenceContext;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::feature_embedding::Dinov2;
using EmbeddingOutput = jinq::models::io_define::feature_embedding::std_feature_embedding_output;

namespace {

NamedTensor f32_tensor(const std::string &name, const std::vector<int64_t> &shape) {
    NamedTensor output;
    output.name = name;
    output.tensor = Tensor::make<float>(shape);
    output.tensor.buffer.assign(output.tensor.buffer.size(), 0);
    return output;
}

template <typename MODEL> class CallablePostprocess : public MODEL {
  public:
    using MODEL::postprocess;
};

template <typename MODEL> class CallableDinov2 : public MODEL {
  public:
    using MODEL::_m_default_pooling;
    using MODEL::_m_supports_mean;
    using MODEL::postprocess;
};

using TestModel = CallablePostprocess<Dinov2<jinq::models::io_define::common_io::mat_input, EmbeddingOutput>>;

} // namespace

TEST(Dinov2FeatureEmbedding, CopiesWholeClsEmbeddingWithoutClassMapping) {
    TestModel model;

    auto output = f32_tensor("cls_tokens", {1, 4});
    auto *data = output.tensor.data<float>();
    data[0] = 0.5f;
    data[1] = -1.5f;
    data[2] = 2.0f;
    data[3] = 0.25f;

    EmbeddingOutput result;
    const InferenceContext empty_context;
    ASSERT_EQ(model.postprocess({output}, empty_context, result), StatusCode::OK);
    ASSERT_EQ(result.embedding.size(), 4u);
    EXPECT_FLOAT_EQ(result.embedding[0], 0.5f);
    EXPECT_FLOAT_EQ(result.embedding[1], -1.5f);
    EXPECT_FLOAT_EQ(result.embedding[2], 2.0f);
    EXPECT_FLOAT_EQ(result.embedding[3], 0.25f);
}

TEST(Dinov2FeatureEmbedding, AcceptsSingleRowRankOneOutput) {
    TestModel model;

    auto output = f32_tensor("cls_tokens", {3});
    auto *data = output.tensor.data<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;

    EmbeddingOutput result;
    const InferenceContext empty_context;
    ASSERT_EQ(model.postprocess({output}, empty_context, result), StatusCode::OK);
    EXPECT_EQ(result.embedding.size(), 3u);
}

TEST(Dinov2FeatureEmbedding, L2NormalizesOnlyWhenRequested) {
    TestModel model;

    auto output = f32_tensor("cls_tokens", {1, 3});
    auto *data = output.tensor.data<float>();
    data[0] = 3.0f;
    data[1] = 0.0f;
    data[2] = 4.0f; // norm = 5

    // default: raw embedding
    EmbeddingOutput raw;
    ASSERT_EQ(model.postprocess({output}, InferenceContext{}, raw), StatusCode::OK);
    EXPECT_FLOAT_EQ(raw.embedding[0], 3.0f);
    EXPECT_FLOAT_EQ(raw.embedding[2], 4.0f);

    // normalize=true: unit L2 norm
    jinq::models::backend::ParamSet params;
    params.set_bool("normalize", true);
    InferenceContext with_params;
    with_params.params = &params;
    EmbeddingOutput normalized;
    ASSERT_EQ(model.postprocess({output}, with_params, normalized), StatusCode::OK);
    double norm = 0.0;
    for (const float value : normalized.embedding) {
        norm += static_cast<double>(value) * static_cast<double>(value);
    }
    EXPECT_NEAR(std::sqrt(norm), 1.0, 1e-6);
    EXPECT_FLOAT_EQ(normalized.embedding[0], 0.6f);
    EXPECT_FLOAT_EQ(normalized.embedding[2], 0.8f);

    // normalize=false: explicit false keeps the raw vector
    params = jinq::models::backend::ParamSet();
    params.set_bool("normalize", false);
    EmbeddingOutput explicit_raw;
    ASSERT_EQ(model.postprocess({output}, with_params, explicit_raw), StatusCode::OK);
    EXPECT_FLOAT_EQ(explicit_raw.embedding[0], 3.0f);
}

TEST(Dinov2FeatureEmbedding, PoolingDefaultsToClsAndAcceptsExplicitCls) {
    TestModel model;

    auto output = f32_tensor("tokens", {1, 2, 3});
    auto *data = output.tensor.data<float>();
    data[0] = 1.0f; // token 0 (the [CLS] token)
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 9.0f; // token 1 (patch token)
    data[4] = 8.0f;
    data[5] = 7.0f;

    // default pooling (cls): the [CLS] token is the first token of the sequence
    EmbeddingOutput cls;
    ASSERT_EQ(model.postprocess({output}, InferenceContext{}, cls), StatusCode::OK);
    ASSERT_EQ(cls.embedding.size(), 3u);
    EXPECT_FLOAT_EQ(cls.embedding[0], 1.0f);
    EXPECT_FLOAT_EQ(cls.embedding[2], 3.0f);

    // explicit pooling=cls request: identical result
    jinq::models::backend::ParamSet params;
    params.set_str("pooling", "cls");
    InferenceContext with_params;
    with_params.params = &params;
    EmbeddingOutput explicit_cls;
    ASSERT_EQ(model.postprocess({output}, with_params, explicit_cls), StatusCode::OK);
    EXPECT_EQ(explicit_cls.embedding, cls.embedding);
}

TEST(Dinov2FeatureEmbedding, MeanPoolingAveragesAllTokens) {
    // the mean branch is gated behind _m_supports_mean, which is set at init
    // from the exported output rank; an all-token export ([1,T,D]) enables it
    CallableDinov2<Dinov2<jinq::models::io_define::common_io::mat_input, EmbeddingOutput>> model;
    model._m_supports_mean = true;
    model._m_default_pooling = "mean";

    auto output = f32_tensor("tokens", {1, 2, 3});
    auto *data = output.tensor.data<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 9.0f;
    data[4] = 8.0f;
    data[5] = 7.0f;

    EmbeddingOutput result;
    ASSERT_EQ(model.postprocess({output}, InferenceContext{}, result), StatusCode::OK);
    ASSERT_EQ(result.embedding.size(), 3u);
    EXPECT_FLOAT_EQ(result.embedding[0], 5.0f); // (1 + 9) / 2
    EXPECT_FLOAT_EQ(result.embedding[1], 5.0f); // (2 + 8) / 2
    EXPECT_FLOAT_EQ(result.embedding[2], 5.0f); // (3 + 7) / 2
}

TEST(Dinov2FeatureEmbedding, MeanPoolingRejectedOnClsOnlyExport) {
    // a rank-2 export ([1,D]) carries only the [CLS] token: mean must fail
    // loudly instead of silently returning a wrong-semantics embedding
    CallableDinov2<Dinov2<jinq::models::io_define::common_io::mat_input, EmbeddingOutput>> model;
    model._m_supports_mean = false;
    model._m_default_pooling = "mean";

    auto output = f32_tensor("cls_tokens", {1, 3});
    EmbeddingOutput result;
    EXPECT_EQ(model.postprocess({output}, InferenceContext{}, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(Dinov2FeatureEmbedding, RejectsMalformedOutputs) {
    TestModel model;
    EmbeddingOutput result;
    const InferenceContext empty_context;

    // empty output list
    EXPECT_EQ(model.postprocess({}, empty_context, result), StatusCode::MODEL_EMPTY_OUTPUT);

    // non-f32 dtype
    NamedTensor i32_output;
    i32_output.name = "cls_tokens";
    i32_output.tensor = Tensor::make<int32_t>({1, 2});
    EXPECT_EQ(model.postprocess({i32_output}, empty_context, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // unexpected rank (neither rank 1, [1, N] nor all-token [1, T, D])
    auto rank4 = f32_tensor("tokens", {1, 2, 2, 2});
    EXPECT_EQ(model.postprocess({rank4}, empty_context, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // rank-2 with a leading dim != 1 (not a single cls embedding)
    auto batch2 = f32_tensor("tokens", {2, 2});
    EXPECT_EQ(model.postprocess({batch2}, empty_context, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // all-token shape with an empty token axis. Tensor::make() refuses a
    // zero-volume shape, so the malformed tensor is assembled by hand.
    NamedTensor no_tokens;
    no_tokens.name = "tokens";
    no_tokens.tensor.dtype = jinq::models::backend::DType::F32;
    no_tokens.tensor.shape = {1, 0, 3};
    EXPECT_EQ(model.postprocess({no_tokens}, empty_context, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // non-finite values
    auto nan_tensor = f32_tensor("cls_tokens", {1, 2});
    nan_tensor.tensor.data<float>()[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(model.postprocess({nan_tensor}, empty_context, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}
