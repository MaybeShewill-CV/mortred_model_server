#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "models/backend/f32_output.h"
#include "models/backend/request_geometry.h"
#include "models/classification/mobilenetv2.h"
#include "models/enhancement/real_esrgan.h"
#include "models/matting/modnet_matting.h"
#include "models/matting/pp_matting.h"
#include "models/model_io_define.h"
#include "models/ocr/db_text_detector.h"
#include "models/scene_segmentation/bisenetv2.h"
#include "models/scene_segmentation/pp_humanseg.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::F32OutputView;
using jinq::models::backend::GeometryScale;
using jinq::models::backend::InferenceContext;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorContract;
using jinq::models::backend::validated_f32_named_output;

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

TEST(RequestGeometry, BuildsNonUniformScaleAndRejectsInvalidContext) {
    const auto request = context(cv::Size(20, 30), cv::Size(10, 5));
    GeometryScale scale;
    std::string error;
    ASSERT_TRUE(jinq::models::backend::make_geometry_scale(request, &scale, &error)) << error;
    EXPECT_FLOAT_EQ(scale.width, 2.0f);
    EXPECT_FLOAT_EQ(scale.height, 6.0f);

    const auto bbox = jinq::models::backend::scale_bbox({1.0f, 2.0f, 3.0f, 4.0f}, scale);
    EXPECT_FLOAT_EQ(bbox.x, 2.0f);
    EXPECT_FLOAT_EQ(bbox.y, 12.0f);
    EXPECT_FLOAT_EQ(bbox.width, 6.0f);
    EXPECT_FLOAT_EQ(bbox.height, 24.0f);

    const auto point = jinq::models::backend::scale_point({1.5f, 2.5f}, scale);
    EXPECT_FLOAT_EQ(point.x, 3.0f);
    EXPECT_FLOAT_EQ(point.y, 15.0f);

    EXPECT_FALSE(jinq::models::backend::make_geometry_scale(context(cv::Size(), cv::Size(10, 10)), &scale, &error));
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
    CallablePostprocess<jinq::models::classification::MobileNetv2<jinq::models::io_define::common_io::mat_input,
                                                                  jinq::models::io_define::classification::std_classification_output>>
        model;
    using Output = jinq::models::io_define::classification::std_classification_output;

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
    CallableInputSize<jinq::models::scene_segmentation::BiseNetV2<
        jinq::models::io_define::common_io::mat_input, jinq::models::io_define::scene_segmentation::std_scene_segmentation_output>>
        bisenet;
    using SegOutput = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
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
}

TEST(ModelOutputContract, BinarySegmentationDecodesNetworkPlanes) {
    CallableInputSize<jinq::models::scene_segmentation::PPHumanSeg<
        jinq::models::io_define::common_io::mat_input, jinq::models::io_define::scene_segmentation::std_scene_segmentation_output>>
        model;
    using SegOutput = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
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
    using MatInput = jinq::models::io_define::common_io::mat_input;
    using MatOutput = jinq::models::io_define::matting::std_matting_output;

    CallableInputSize<jinq::models::matting::ModNetMatting<MatInput, MatOutput>> modnet;
    CallableInputSize<jinq::models::matting::PPMatting<MatInput, MatOutput>> ppmatting;
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
    EXPECT_EQ(ppmatting_result.matting_result.type(), CV_32SC1);

    alpha.tensor.shape = {1, 1, 3, 2};
    EXPECT_EQ(modnet.postprocess({alpha}, request, modnet_result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_EQ(ppmatting.postprocess({alpha}, request, ppmatting_result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(ModelOutputContract, OcrValidatesProbabilityMapAndMapsGeometry) {
    CallableOcrFields<jinq::models::ocr::DBTextDetector<jinq::models::io_define::common_io::mat_input,
                                                        jinq::models::io_define::ocr::std_text_regions_output>>
        model;
    using OcrOutput = jinq::models::io_define::ocr::std_text_regions_output;
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
    CallablePostprocess<jinq::models::enhancement::RealEsrGan<jinq::models::io_define::common_io::mat_input,
                                                              jinq::models::io_define::enhancement::std_enhancement_output>>
        model;
    using EnhOutput = jinq::models::io_define::enhancement::std_enhancement_output;

    auto image = f32_tensor("output", {1, 3, 4, 6}, 0.5f);
    EnhOutput result;
    const InferenceContext request;
    ASSERT_EQ(model.postprocess({image}, request, result), StatusCode::OK);
    EXPECT_EQ(result.enhancement_result.size(), cv::Size(6, 4));
    EXPECT_EQ(result.enhancement_result.type(), CV_8UC3);

    image.tensor.shape = {1, 4, 4, 6};
    EXPECT_EQ(model.postprocess({image}, request, result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}
