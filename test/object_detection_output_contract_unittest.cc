#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "models/backend/tensor.h"
#include "models/model_io_define.h"
#include "models/object_detection/yolov8_detector.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::NamedTensor;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::object_detection::YoloV8Detector;

namespace {

class TestYoloV8Detector : public YoloV8Detector<mat_input, std_object_detection_output> {
  public:
    using YoloV8Detector<mat_input, std_object_detection_output>::postprocess;
    using YoloV8Detector<mat_input, std_object_detection_output>::_m_detection_params;
};

NamedTensor output_tensor(DType dtype, const std::vector<int64_t> &shape, std::vector<float> values) {
    NamedTensor output;
    output.name = "output0";
    output.tensor.dtype = dtype;
    output.tensor.shape = shape;
    const size_t bytes = values.size() * sizeof(float);
    output.tensor.buffer.resize(bytes);
    if (!values.empty()) {
        std::memcpy(output.tensor.buffer.data(), values.data(), bytes);
    }
    return output;
}

std::vector<float> one_candidate_yolov8(float class_score) {
    std::vector<float> values(84, 0.0f);
    values[0] = 10.0f;
    values[1] = 10.0f;
    values[2] = 5.0f;
    values[3] = 5.0f;
    values[4 + 5] = class_score;
    return values;
}

jinq::models::InferenceContext test_context() {
    jinq::models::InferenceContext context;
    context.source_size = cv::Size(20, 20);
    context.network_size = cv::Size(10, 10);
    return context;
}

} // namespace

TEST(ObjectDetectionOutputContract, YoloV8AcceptsValidOutputAndMapsGeometry) {
    TestYoloV8Detector detector;
    detector._m_detection_params.class_nums = 80;
    detector._m_detection_params.score_threshold = 0.25f;
    detector._m_detection_params.nms_threshold = 0.5f;
    detector._m_detection_params.keep_top_k = 100;
    detector._m_detection_params.min_box_area_px = 0.0f;

    std::vector<NamedTensor> outputs;
    outputs.push_back(output_tensor(DType::F32, {1, 84, 1}, one_candidate_yolov8(0.9f)));
    std_object_detection_output result;
    ASSERT_EQ(detector.postprocess(outputs, test_context(), result), StatusCode::OK);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].class_id, 5);
    EXPECT_NEAR(result[0].score, 0.9f, 1e-6);
    EXPECT_NEAR(result[0].bbox.x, 15.0f, 1e-5);
    EXPECT_NEAR(result[0].bbox.y, 15.0f, 1e-5);
    EXPECT_NEAR(result[0].bbox.width, 10.0f, 1e-5);
    EXPECT_NEAR(result[0].bbox.height, 10.0f, 1e-5);
}

TEST(ObjectDetectionOutputContract, YoloV8RejectsMalformedOutput) {
    TestYoloV8Detector detector;
    detector._m_detection_params.class_nums = 80;
    detector._m_detection_params.score_threshold = 0.25f;

    const auto check = [&detector](std::vector<NamedTensor> outputs) {
        std_object_detection_output result;
        EXPECT_EQ(detector.postprocess(outputs, test_context(), result), StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
        EXPECT_TRUE(result.empty());
    };

    check({output_tensor(DType::I32, {1, 84, 1}, one_candidate_yolov8(0.9f))});
    check({output_tensor(DType::F32, {1, 83, 1}, one_candidate_yolov8(0.9f))});

    auto short_buffer = output_tensor(DType::F32, {1, 84, 1}, one_candidate_yolov8(0.9f));
    short_buffer.tensor.buffer.pop_back();
    check({short_buffer});

    auto nan_output = output_tensor(DType::F32, {1, 84, 1}, one_candidate_yolov8(std::numeric_limits<float>::quiet_NaN()));
    check({nan_output});
}
