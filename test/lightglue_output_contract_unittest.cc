#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "models/backend/tensor.h"
#include "models/feature_point/lightglue.h"
#include "models/io/common_input.h"
#include "models/io/feature_point.h"

using mortred::common::StatusCode;
using mortred::models::backend::NamedTensor;
using mortred::models::backend::Tensor;
using Model = mortred::models::feature_point::LightGlue<mortred::models::io_define::common_io::pair_mat_input,
                                                     mortred::models::io_define::feature_point::std_feature_point_match_output>;

namespace {

template <typename T> class CallableLightGlue : public T {
  public:
    using T::classify_extractor_outputs;
    using T::classify_feature_layout;
    using T::classify_matcher_outputs;
    using T::FeaturePoints;
};

using TestModel = CallableLightGlue<Model>;

NamedTensor make_tensor(const std::string &name, Tensor tensor) {
    NamedTensor named;
    named.name = name;
    named.tensor = std::move(tensor);
    return named;
}

Tensor legal_keypoints() { return Tensor::make<int32_t>({2}); }

Tensor legal_scores() {
    auto tensor = Tensor::make<float>({1});
    tensor.data<float>()[0] = 1.0f;
    return tensor;
}

Tensor legal_descriptors() { return Tensor::make<float>({256}); }

std::vector<NamedTensor> extractor_outputs(Tensor keypoints, Tensor scores, Tensor descriptors) {
    return {
        make_tensor("keypoints", std::move(keypoints)),
        make_tensor("scores", std::move(scores)),
        make_tensor("descriptors", std::move(descriptors)),
    };
}

Tensor legal_matches() { return Tensor::make<int32_t>({2}); }

Tensor legal_match_scores() {
    auto tensor = Tensor::make<float>({1});
    tensor.data<float>()[0] = 1.0f;
    return tensor;
}

std::vector<NamedTensor> matcher_outputs(Tensor matches, Tensor scores) {
    return {
        make_tensor("matches0", std::move(matches)),
        make_tensor("mscores0", std::move(scores)),
    };
}

void expect_contract_failed(StatusCode status) {
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

} // namespace

TEST(LightGlueOutputContract, MissingExtractorOutputsStayEmpty) {
    TestModel model;
    const auto status = model.classify_extractor_outputs({
        make_tensor("scores", Tensor::make<float>({1})),
        make_tensor("descriptors", Tensor::make<float>({256})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(LightGlueOutputContract, ExtractorDtypeMismatchReturnsContractFailed) {
    TestModel model;
    const auto status = model.classify_extractor_outputs({
        make_tensor("keypoints", Tensor::make<float>({2})),
        make_tensor("scores", Tensor::make<float>({1})),
        make_tensor("descriptors", Tensor::make<float>({256})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(LightGlueOutputContract, ExtractorSizeMismatchReturnsContractFailed) {
    TestModel model;
    const auto status = model.classify_extractor_outputs({
        make_tensor("keypoints", Tensor::make<int32_t>({2})),
        make_tensor("scores", Tensor::make<float>({1})),
        make_tensor("descriptors", Tensor::make<float>({128})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(LightGlueOutputContract, MissingMatcherOutputsStayEmpty) {
    TestModel model;
    const auto status = model.classify_matcher_outputs({make_tensor("mscores0", Tensor::make<float>({1}))});
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(LightGlueOutputContract, FeatureLayoutMismatchReturnsContractFailed) {
    TestModel model;
    TestModel::FeaturePoints src;
    src.keypoints = {1.0f, 2.0f};
    src.normalized_keypoints = {0.0f, 0.0f};
    src.descriptors = {0.0f};
    TestModel::FeaturePoints dst = src;
    const auto status = model.classify_feature_layout(src, dst);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(LightGlueOutputContract, ExtractorScoresWrongDtype) {
    TestModel model;
    const auto status =
        model.classify_extractor_outputs(extractor_outputs(legal_keypoints(), Tensor::make<int32_t>({1}), legal_descriptors()));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, ExtractorDescriptorsWrongDtype) {
    TestModel model;
    const auto status =
        model.classify_extractor_outputs(extractor_outputs(legal_keypoints(), legal_scores(), Tensor::make<int32_t>({256})));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, ExtractorScoresNan) {
    TestModel model;
    auto scores = legal_scores();
    scores.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    const auto status = model.classify_extractor_outputs(extractor_outputs(legal_keypoints(), std::move(scores), legal_descriptors()));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, ExtractorDescriptorsNan) {
    TestModel model;
    auto descriptors = legal_descriptors();
    descriptors.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    const auto status = model.classify_extractor_outputs(extractor_outputs(legal_keypoints(), legal_scores(), std::move(descriptors)));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, ExtractorScoresPositiveInf) {
    TestModel model;
    auto scores = legal_scores();
    scores.data<float>()[0] = std::numeric_limits<float>::infinity();
    const auto status = model.classify_extractor_outputs(extractor_outputs(legal_keypoints(), std::move(scores), legal_descriptors()));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, ExtractorDescriptorsPositiveInf) {
    TestModel model;
    auto descriptors = legal_descriptors();
    descriptors.data<float>()[0] = std::numeric_limits<float>::infinity();
    const auto status = model.classify_extractor_outputs(extractor_outputs(legal_keypoints(), legal_scores(), std::move(descriptors)));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, MatcherMatchesWrongDtype) {
    TestModel model;
    const auto status = model.classify_matcher_outputs(matcher_outputs(Tensor::make<float>({2}), legal_match_scores()));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, MatcherScoresWrongDtype) {
    TestModel model;
    const auto status = model.classify_matcher_outputs(matcher_outputs(legal_matches(), Tensor::make<int32_t>({1})));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, MatcherScoresNan) {
    TestModel model;
    auto scores = legal_match_scores();
    scores.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    const auto status = model.classify_matcher_outputs(matcher_outputs(legal_matches(), std::move(scores)));
    expect_contract_failed(status);
}

TEST(LightGlueOutputContract, MatcherScoresPositiveInf) {
    TestModel model;
    auto scores = legal_match_scores();
    scores.data<float>()[0] = std::numeric_limits<float>::infinity();
    const auto status = model.classify_matcher_outputs(matcher_outputs(legal_matches(), std::move(scores)));
    expect_contract_failed(status);
}
