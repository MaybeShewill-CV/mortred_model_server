#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "models/backend/f32_output.h"
#include "models/backend/param_spec.h"
#include "models/backend/request_geometry.h"
#include "models/classification/mobilenetv2.h"
#include "models/enhancement/real_esrgan.h"
#include "models/matting/modnet_matting.h"
#include "models/matting/pp_matting.h"
#include "models/model_io_define.h"
#include "models/ocr/db_text_detector.h"
#include "models/scene_segmentation/bisenetv2.h"
#include "models/scene_segmentation/pp_humanseg.h"

using mortred::common::StatusCode;
using mortred::models::backend::DType;
using mortred::models::backend::F32OutputView;
using mortred::models::backend::GeometryScale;
using mortred::models::backend::InferenceContext;
using mortred::models::backend::NamedTensor;
using mortred::models::backend::Tensor;
using mortred::models::backend::TensorContract;
using mortred::models::backend::validated_f32_named_output;

namespace {

NamedTensor f32_tensor(const std::string &name, const std::vector<int64_t> &shape, float fill = 0.0f) {
    NamedTensor output;
    output.name = name;
    output.tensor = Tensor::make<float>(shape);
    output.tensor.buffer.assign(output.tensor.buffer.size(), 0);
    auto *data = reinterpret_cast<float *>(output.tensor.buffer.data());
    for (int64_t idx = 0; idx < output.tensor.element_count(); ++idx) {
        data[idx] = fill;
    }
    return output;
}

InferenceContext context(const cv::Size &source, const cv::Size &network) {
    InferenceContext result;
    result.source_size = source;
    result.network_size = network;
    return result;
}

template <typename MODEL> class CallablePostprocess : public MODEL {
  public:
    using MODEL::postprocess;
};

template <typename MODEL> class CallableInputSize : public MODEL {
  public:
    using MODEL::_m_input_size_host;
    using MODEL::postprocess;
};

template <typename MODEL> class CallableOcrFields : public MODEL {
  public:
    using MODEL::_m_input_size_host;
    using MODEL::_m_output_name;
    using MODEL::postprocess;
};

} // namespace

TEST(ModelOutputContract, ClassificationTopKKeepsHighestScoresDescending) {
    CallablePostprocess<mortred::models::classification::MobileNetv2<mortred::models::io_define::common_io::mat_input,
                                                                 mortred::models::io_define::classification::std_classification_output>>
        model;
    using ClsOutput = mortred::models::io_define::classification::std_classification_output;

    // scores: [0.1, 0.7, 0.2, 0.6, 0.3] -> top 2 = {0.7, 0.6}, argmax = 1
    auto scores = f32_tensor("scores", {1, 5}, 0.1f);
    auto *values = scores.tensor.data<float>();
    values[1] = 0.7f;
    values[2] = 0.2f;
    values[3] = 0.6f;
    values[4] = 0.3f;

    const InferenceContext request;

    // legacy path: full class-index ordered array
    ClsOutput full;
    ASSERT_EQ(model.postprocess({scores}, request, full), StatusCode::OK);
    ASSERT_EQ(full.scores.size(), 5u);
    EXPECT_EQ(full.class_id, 1);

    // top_k = 2: two highest scores, descending
    mortred::models::backend::ParamSet params;
    params.set_i32("top_k", 2);
    auto with_top_k = request;
    with_top_k.params = &params;
    ClsOutput top;
    ASSERT_EQ(model.postprocess({scores}, with_top_k, top), StatusCode::OK);
    ASSERT_EQ(top.scores.size(), 2u);
    EXPECT_FLOAT_EQ(top.scores[0], 0.7f);
    EXPECT_FLOAT_EQ(top.scores[1], 0.6f);
    EXPECT_EQ(top.class_id, 1);  // argmax unchanged by truncation

    // top_k larger than the class count keeps everything
    params = mortred::models::backend::ParamSet();
    params.set_i32("top_k", 100);
    with_top_k.params = &params;
    ClsOutput all;
    ASSERT_EQ(model.postprocess({scores}, with_top_k, all), StatusCode::OK);
    EXPECT_EQ(all.scores.size(), 5u);
}

TEST(RequestGeometry, BuildsNonUniformScaleAndRejectsInvalidContext) {
    const auto request = context(cv::Size(20, 30), cv::Size(10, 5));
    GeometryScale scale;
    std::string error;
    ASSERT_TRUE(mortred::models::backend::make_geometry_scale(request, &scale, &error)) << error;
    EXPECT_FLOAT_EQ(scale.width, 2.0f);
    EXPECT_FLOAT_EQ(scale.height, 6.0f);

    const auto bbox = mortred::models::backend::scale_bbox({1.0f, 2.0f, 3.0f, 4.0f}, scale);
    EXPECT_FLOAT_EQ(bbox.x, 2.0f);
    EXPECT_FLOAT_EQ(bbox.y, 12.0f);
    EXPECT_FLOAT_EQ(bbox.width, 6.0f);
    EXPECT_FLOAT_EQ(bbox.height, 24.0f);

    const auto point = mortred::models::backend::scale_point({1.5f, 2.5f}, scale);
    EXPECT_FLOAT_EQ(point.x, 3.0f);
    EXPECT_FLOAT_EQ(point.y, 15.0f);

    EXPECT_FALSE(mortred::models::backend::make_geometry_scale(context(cv::Size(), cv::Size(10, 10)), &scale, &error));
    EXPECT_NE(error.find("invalid request geometry"), std::string::npos);
}

TEST(F32OutputContract, ValidatesNameShapeBufferAndFiniteValues) {
    std::vector<NamedTensor> outputs{f32_tensor("scores", {1, 3})};
    F32OutputView view;
    const auto contract = TensorContract{DType::F32, 2, {1, 3}};
    ASSERT_EQ(validated_f32_named_output(outputs, "scores", contract, "test model", &view), StatusCode::OK);
    ASSERT_NE(view.tensor, nullptr);
    ASSERT_NE(view.data, nullptr);
    EXPECT_EQ(view.tensor->shape, std::vector<int64_t>({1, 3}));

    EXPECT_EQ(validated_f32_named_output(outputs, "missing", contract, "test model"), StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_EQ(validated_f32_named_output(outputs, "scores", TensorContract{DType::F32, 2, {1, 4}}, "test model"),
              StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    outputs[0].tensor.buffer.pop_back();
    EXPECT_EQ(validated_f32_named_output(outputs, "scores", contract, "test model"), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    outputs[0] = f32_tensor("scores", {1, 3});
    reinterpret_cast<float *>(outputs[0].tensor.buffer.data())[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(validated_f32_named_output(outputs, "scores", contract, "test model"), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ModelOutputContract, ClassificationAcceptsSingleAndBatchItemScores) {
    CallablePostprocess<mortred::models::classification::MobileNetv2<mortred::models::io_define::common_io::mat_input,
                                                                  mortred::models::io_define::classification::std_classification_output>>
        model;
    using Output = mortred::models::io_define::classification::std_classification_output;

    const auto run = [&model](const std::vector<int64_t> &shape, std::vector<float> values, StatusCode expected) {
        std::vector<NamedTensor> outputs;
        outputs.push_back(f32_tensor("scores", shape));
        auto *data = reinterpret_cast<float *>(outputs.front().tensor.buffer.data());
        std::copy(values.begin(), values.end(), data);
        Output result;
        const InferenceContext empty_context;
        EXPECT_EQ(model.postprocess(outputs, empty_context, result), expected);
        if (expected == StatusCode::OK) {
            EXPECT_EQ(result.scores.size(), values.size());
            EXPECT_EQ(result.class_id, 2);
        }
    };

    run({1, 3}, {0.1f, 0.2f, 0.7f}, StatusCode::OK);
    run({3}, {0.1f, 0.2f, 0.7f}, StatusCode::OK);
    run({2, 2}, {0.1f, 0.2f, 0.3f}, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ModelOutputContract, SceneSegmentationMapsToRequestSourceSize) {
    CallableInputSize<mortred::models::scene_segmentation::BiseNetV2<
        mortred::models::io_define::common_io::mat_input, mortred::models::io_define::scene_segmentation::std_scene_segmentation_output>>
        bisenet;
    using SegOutput = mortred::models::io_define::scene_segmentation::std_scene_segmentation_output;
    bisenet._m_input_size_host = cv::Size(3, 2);

    std::vector<NamedTensor> outputs{f32_tensor("final_output", {2, 3, 2})};
    auto *data = reinterpret_cast<float *>(outputs.front().tensor.buffer.data());
    for (size_t idx = 0; idx < 6; ++idx) {
        data[idx * 2] = -1.0f;
        data[idx * 2 + 1] = 1.0f;
    }
    SegOutput result;
    ASSERT_EQ(bisenet.postprocess(outputs, context(cv::Size(9, 6), cv::Size(3, 2)), result), StatusCode::OK);
    EXPECT_EQ(result.segmentation_result.size(), cv::Size(9, 6));
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 0), 1);

    outputs.front().tensor.shape = {2, 2, 2};
    EXPECT_EQ(bisenet.postprocess(outputs, context(cv::Size(9, 6), cv::Size(3, 2)), result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    // NCHW [1,C,H,W] after MNN host conversion: class 1 wins every pixel
    std::vector<NamedTensor> nchw_outputs{f32_tensor("final_output", {1, 2, 2, 3})};
    auto *nchw_data = reinterpret_cast<float *>(nchw_outputs.front().tensor.buffer.data());
    const auto plane = 2 * 3;
    for (int idx = 0; idx < plane; ++idx) {
        nchw_data[idx] = -1.0f;
        nchw_data[plane + idx] = 1.0f;
    }
    ASSERT_EQ(bisenet.postprocess(nchw_outputs, context(cv::Size(9, 6), cv::Size(3, 2)), result), StatusCode::OK);
    EXPECT_EQ(result.segmentation_result.size(), cv::Size(9, 6));
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 0), 1);
}

TEST(ModelOutputContract, BinarySegmentationDecodesNetworkPlanes) {
    CallableInputSize<mortred::models::scene_segmentation::PPHumanSeg<
        mortred::models::io_define::common_io::mat_input, mortred::models::io_define::scene_segmentation::std_scene_segmentation_output>>
        model;
    using SegOutput = mortred::models::io_define::scene_segmentation::std_scene_segmentation_output;
    model._m_input_size_host = cv::Size(3, 2);

    auto output = f32_tensor("softmax", {1, 2, 2, 3});
    auto *data = reinterpret_cast<float *>(output.tensor.buffer.data());
    const size_t plane = 6;
    for (size_t idx = 0; idx < plane; ++idx) {
        data[idx] = -1.0f;
        data[plane + idx] = 1.0f;
    }
    SegOutput result;
    const auto request = context(cv::Size(12, 8), cv::Size(3, 2));
    ASSERT_EQ(model.postprocess({output}, request, result), StatusCode::OK);
    EXPECT_EQ(result.segmentation_result.size(), cv::Size(12, 8));
    EXPECT_EQ(result.segmentation_result.at<int32_t>(0, 0), 1);

    output.tensor.shape = {1, 3, 2, 3};
    EXPECT_EQ(model.postprocess({output}, request, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ModelOutputContract, MattingResizesAlphaToRequestSourceSize) {
    using MatInput = mortred::models::io_define::common_io::mat_input;
    using MatOutput = mortred::models::io_define::matting::std_matting_output;

    CallableInputSize<mortred::models::matting::ModNetMatting<MatInput, MatOutput>> modnet;
    CallableInputSize<mortred::models::matting::PPMatting<MatInput, MatOutput>> ppmatting;
    modnet._m_input_size_host = cv::Size(3, 2);
    ppmatting._m_input_size_host = cv::Size(3, 2);

    auto alpha = f32_tensor("alpha", {1, 1, 2, 3}, 0.5f);
    MatOutput modnet_result;
    const auto request = context(cv::Size(12, 8), cv::Size(3, 2));
    ASSERT_EQ(modnet.postprocess({alpha}, request, modnet_result), StatusCode::OK);
    EXPECT_EQ(modnet_result.matting_result.size(), cv::Size(12, 8));
    EXPECT_EQ(modnet_result.matting_result.type(), CV_8UC1);

    MatOutput ppmatting_result;
    ASSERT_EQ(ppmatting.postprocess({alpha}, request, ppmatting_result), StatusCode::OK);
    EXPECT_EQ(ppmatting_result.matting_result.size(), cv::Size(12, 8));
    // ac515586 aligned the matting tails: pp_matting now resizes bilinearly and
    // emits a CV_8UC1 matte, matching modnet (it used to be CV_32SC1, which made
    // the two matting models expose different alpha semantics to callers).
    EXPECT_EQ(ppmatting_result.matting_result.type(), CV_8UC1);

    alpha.tensor.shape = {1, 1, 3, 2};
    EXPECT_EQ(modnet.postprocess({alpha}, request, modnet_result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_EQ(ppmatting.postprocess({alpha}, request, ppmatting_result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ModelOutputContract, OcrValidatesProbabilityMapAndMapsGeometry) {
    CallableOcrFields<mortred::models::ocr::DBTextDetector<mortred::models::io_define::common_io::mat_input,
                                                        mortred::models::io_define::ocr::std_text_regions_output>>
        model;
    using OcrOutput = mortred::models::io_define::ocr::std_text_regions_output;
    model._m_output_name = "prob";
    model._m_input_size_host = cv::Size(3, 2);

    auto prob = f32_tensor("prob", {1, 1, 2, 3});
    OcrOutput result;
    const auto request = context(cv::Size(9, 6), cv::Size(3, 2));
    ASSERT_EQ(model.postprocess({prob}, request, result), StatusCode::OK);
    EXPECT_TRUE(result.empty());

    prob.name = "wrong";
    EXPECT_EQ(model.postprocess({prob}, request, result), StatusCode::MODEL_EMPTY_OUTPUT);
    prob.name = "prob";
    prob.tensor.shape = {1, 1, 3, 2};
    EXPECT_EQ(model.postprocess({prob}, request, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ModelOutputContract, EnhancementValidatesSuperResolutionOutput) {
    CallablePostprocess<mortred::models::enhancement::RealEsrGan<mortred::models::io_define::common_io::mat_input,
                                                              mortred::models::io_define::enhancement::std_enhancement_output>>
        model;
    using EnhOutput = mortred::models::io_define::enhancement::std_enhancement_output;

    auto image = f32_tensor("output", {1, 3, 4, 6}, 0.5f);
    EnhOutput result;
    const InferenceContext request;
    ASSERT_EQ(model.postprocess({image}, request, result), StatusCode::OK);
    EXPECT_EQ(result.enhancement_result.size(), cv::Size(6, 4));
    EXPECT_EQ(result.enhancement_result.type(), CV_8UC3);

    image.tensor.shape = {1, 4, 4, 6};
    EXPECT_EQ(model.postprocess({image}, request, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ModelOutputContract, OcrScoreThresholdSweepIsMonotone) {
    CallableOcrFields<mortred::models::ocr::DBTextDetector<mortred::models::io_define::common_io::mat_input,
                                                         mortred::models::io_define::ocr::std_text_regions_output>>
        model;
    model._m_output_name = "sigmoid_0.tmp_0";
    using OcrOutput = mortred::models::io_define::ocr::std_text_regions_output;

    // 8x8 prob map: a 4x4 bright block (0.9) on a dark background (0.05);
    // network 8x8 == source 8x8 so the geometry scale is 1:1
    const auto request = context(cv::Size(8, 8), cv::Size(8, 8));
    auto prob = f32_tensor("sigmoid_0.tmp_0", {1, 1, 8, 8}, 0.05f);
    auto *values = prob.tensor.data<float>();
    for (int row = 2; row < 6; ++row) {
        for (int col = 2; col < 6; ++col) {
            values[row * 8 + col] = 0.9f;
        }
    }

    auto run_with_threshold = [&model, &prob, &request](float request_threshold) {
        OcrOutput out;
        auto ctx = request;
        mortred::models::backend::ParamSet params;
        params.set_f32("score_threshold", request_threshold);
        ctx.params = &params;
        const auto status = model.postprocess({prob}, ctx, out);
        EXPECT_EQ(status, StatusCode::OK) << "threshold=" << request_threshold;
        return out.size();
    };

    // the block survives a permissive threshold and vanishes under a strict one
    EXPECT_EQ(run_with_threshold(0.1f), 1u);
    EXPECT_EQ(run_with_threshold(0.95f), 0u);

    // legacy path: the config default (0.4 from DetectionParams-style init)
    // keeps the block because its mean score is well above 0.4
    OcrOutput legacy;
    EXPECT_EQ(model.postprocess({prob}, request, legacy), StatusCode::OK);
    EXPECT_EQ(legacy.size(), 1u);
}
