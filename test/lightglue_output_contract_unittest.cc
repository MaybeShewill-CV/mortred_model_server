#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "models/backend/tensor.h"
#include "models/feature_point/lightglue.h"
#include "models/io/common_input.h"
#include "models/io/feature_point.h"

using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using Model = jinq::models::feature_point::LightGlue<jinq::models::io_define::common_io::pair_mat_input,
                                                     jinq::models::io_define::feature_point::std_feature_point_match_output>;

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
