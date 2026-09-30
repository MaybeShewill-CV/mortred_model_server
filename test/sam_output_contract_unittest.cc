#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "models/backend/tensor.h"
#include "models/segment_anything/sam_prediction/sam_prompt_decoder.h"
#include "models/segment_anything/sam_prediction/sam_vit_encoder.h"

using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;

namespace {

template <typename T> class CallableSamVitEncoder : public T {
  public:
    using T::classify_encoder_outputs;
};

template <typename T> class CallableSamPromptDecoder : public T {
  public:
    using T::classify_decoder_outputs;
    using T::classify_full_mask_tensor;
    using T::classify_low_res_mask_tensor;
};

using TestEncoder = CallableSamVitEncoder<jinq::models::segment_anything::SamVitEncoder>;
using TestDecoder = CallableSamPromptDecoder<jinq::models::segment_anything::SamPromptDecoder>;

NamedTensor make_named(const std::string &name, Tensor tensor) {
    NamedTensor named;
    named.name = name;
    named.tensor = std::move(tensor);
    return named;
}

NamedTensor zero_element_f32(const std::string &name) {
    NamedTensor named;
    named.name = name;
    named.tensor.dtype = jinq::models::backend::DType::F32;
    named.tensor.shape = {1, 0, 2, 2};
    return named;
}

} // namespace

TEST(SamOutputContract, EncoderEmptyOutputsStayEmpty) {
    const auto status = TestEncoder::classify_encoder_outputs({});
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(SamOutputContract, EncoderDtypeMismatchReturnsContractFailed) {
    const auto status = TestEncoder::classify_encoder_outputs({
        make_named("image_embeddings", Tensor::make<int32_t>({1, 256, 64, 64})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(SamOutputContract, EncoderZeroElementCountReturnsContractFailed) {
    const auto status = TestEncoder::classify_encoder_outputs({zero_element_f32("image_embeddings")});
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(SamOutputContract, DecoderFullMaskIllegalRankReturnsContractFailed) {
    const auto status = TestDecoder::classify_decoder_outputs({
        make_named("iou_predictions", Tensor::make<float>({1, 3})),
        make_named("masks", Tensor::make<float>({8, 8})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(SamOutputContract, DecoderLowResMaskIllegalRankReturnsContractFailed) {
    const auto status = TestDecoder::classify_decoder_outputs({
        make_named("iou_predictions", Tensor::make<float>({1, 3})),
        make_named("low_res_masks", Tensor::make<float>({1, 3, 64})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(SamOutputContract, DecoderMissingIouStaysEmpty) {
    const auto status = TestDecoder::classify_decoder_outputs({
        make_named("masks", Tensor::make<float>({1, 3, 8, 8})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
}

TEST(SamOutputContract, DecoderMissingMasksAndLowResStaysEmpty) {
    const auto status = TestDecoder::classify_decoder_outputs({
        make_named("iou_predictions", Tensor::make<float>({1, 3})),
    });
    EXPECT_EQ(status, StatusCode::MODEL_EMPTY_OUTPUT);
}
