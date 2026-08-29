#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "models/model_io_define.h"
#include "models/object_detection/detector_common.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::object_detection::DetectionGeometryScale;
using jinq::models::object_detection::DetectionParams;
using jinq::models::object_detection::F32OutputView;
using jinq::models::object_detection::make_detection_geometry_scale;
using jinq::models::object_detection::make_nchw_input;
using jinq::models::object_detection::scale_detection_bbox;
using jinq::models::object_detection::scale_detection_point;
using jinq::models::object_detection::validated_f32_output;

namespace {

jinq::models::backend::InferenceContext test_context() {
    jinq::models::backend::InferenceContext context;
    context.source_size = cv::Size(200, 300);
    context.network_size = cv::Size(100, 50);
    return context;
}

NamedTensor valid_output() {
    NamedTensor output;
    output.name = "output";
    output.tensor = Tensor::make<float>({1, 84, 1});
    return output;
}

} // namespace

TEST(DetectorCommon, BuildsAndAppliesRequestGeometryScale) {
    const auto context = test_context();
    DetectionGeometryScale scale;
    std::string error;
    ASSERT_TRUE(make_detection_geometry_scale(context, &scale, &error)) << error;
    EXPECT_FLOAT_EQ(scale.width, 2.0f);
    EXPECT_FLOAT_EQ(scale.height, 6.0f);

    const auto bbox = scale_detection_bbox({1.0f, 2.0f, 3.0f, 4.0f}, scale);
    EXPECT_FLOAT_EQ(bbox.x, 2.0f);
    EXPECT_FLOAT_EQ(bbox.y, 12.0f);
    EXPECT_FLOAT_EQ(bbox.width, 6.0f);
    EXPECT_FLOAT_EQ(bbox.height, 24.0f);

    const auto point = scale_detection_point({1.5f, 2.5f}, scale);
    EXPECT_FLOAT_EQ(point.x, 3.0f);
    EXPECT_FLOAT_EQ(point.y, 15.0f);
}

TEST(DetectorCommon, RejectsInvalidRequestGeometry) {
    auto context = test_context();
    context.network_size = cv::Size();
    DetectionGeometryScale scale;
    std::string error;
    EXPECT_FALSE(make_detection_geometry_scale(context, &scale, &error));
    EXPECT_NE(error.find("invalid request geometry"), std::string::npos);
}

TEST(DetectorCommon, ValidatesNamedF32Output) {
    std::vector<NamedTensor> outputs{valid_output()};
    F32OutputView view;
    const auto status = validated_f32_output(outputs, "output", {DType::F32, 3, {1, 84, 1}}, "test detector", &view);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_NE(view.tensor, nullptr);
    ASSERT_NE(view.data, nullptr);
    EXPECT_EQ(view.tensor->shape, std::vector<int64_t>({1, 84, 1}));
    EXPECT_EQ(view.data[0], 0.0f);
}

TEST(DetectorCommon, DistinguishesMissingAndContractFailedOutputs) {
    std::vector<NamedTensor> outputs{valid_output()};
    EXPECT_EQ(validated_f32_output(outputs, "missing", {DType::F32, 1, {1}}, "test detector"), StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_EQ(validated_f32_output(outputs, "output", {DType::F32, 3, {1, 83, 1}}, "test detector"),
              StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    auto non_finite = valid_output();
    *reinterpret_cast<float *>(non_finite.tensor.buffer.data()) = std::numeric_limits<float>::quiet_NaN();
    outputs[0] = non_finite;
    EXPECT_EQ(validated_f32_output(outputs, "output", {DType::F32, 3, {1, 84, 1}}, "test detector"),
              StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
}

TEST(DetectorCommon, FinalizeDetectionsAppliesNmsTopKAndCategories) {
    using jinq::models::io_define::object_detection::bbox;

    DetectionParams params;
    params.class_nums = 2;
    params.class_names = {"person", "car"};
    params.score_threshold = 0.5f;
    params.nms_threshold = 0.9f;
    params.keep_top_k = 1;

    std::vector<bbox> detections;
    bbox first;
    first.bbox = cv::Rect2f(0.0f, 0.0f, 10.0f, 10.0f);
    first.score = 0.9f;
    first.class_id = 0;
    detections.push_back(first);

    bbox duplicate = first;
    duplicate.score = 0.8f;
    detections.push_back(duplicate);

    bbox car = first;
    car.score = 0.7f;
    car.class_id = 1;
    detections.push_back(car);

    const auto result = jinq::models::object_detection::finalize_detections(detections, params);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].class_id, 0);
    EXPECT_FLOAT_EQ(result[0].score, 0.9f);
    EXPECT_EQ(result[0].category, "person");
}

TEST(DetectorCommon, PacksCV32FC3MatAsNchwF32Tensor) {
    cv::Mat image(2, 3, CV_32FC3, cv::Scalar(1.0f, 2.0f, 3.0f));
    NamedTensor input;
    ASSERT_TRUE(make_nchw_input("images", image, &input));
    EXPECT_EQ(input.name, "images");
    EXPECT_EQ(input.tensor.dtype, DType::F32);
    ASSERT_EQ(input.tensor.shape, std::vector<int64_t>({1, 3, 2, 3}));
    ASSERT_EQ(input.tensor.buffer.size(), 18u * sizeof(float));

    const auto *data = reinterpret_cast<const float *>(input.tensor.buffer.data());
    const size_t plane = 6;
    for (size_t idx = 0; idx < plane; ++idx) {
        EXPECT_FLOAT_EQ(data[idx], 1.0f);
        EXPECT_FLOAT_EQ(data[plane + idx], 2.0f);
        EXPECT_FLOAT_EQ(data[2 * plane + idx], 3.0f);
    }
}

TEST(DetectorCommon, RejectsInvalidNchwInput) {
    NamedTensor input;
    EXPECT_FALSE(make_nchw_input("", cv::Mat(1, 1, CV_32FC3), &input));
    EXPECT_FALSE(make_nchw_input("images", cv::Mat(), &input));
    EXPECT_FALSE(make_nchw_input("images", cv::Mat(1, 1, CV_8UC3), &input));
    EXPECT_FALSE(make_nchw_input("images", cv::Mat(1, 1, CV_32FC3), nullptr));
}
