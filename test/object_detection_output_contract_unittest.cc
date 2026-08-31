#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include "models/backend/param_spec.h"

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

jinq::models::backend::InferenceContext test_context() {
    jinq::models::backend::InferenceContext context;
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

TEST(ObjectDetectionOutputContract, RequestParamThresholdSweepIsMonotone) {
    // three candidates with distinct scores: 0.2 / 0.5 / 0.9
    // [1, 84, N] is channel-major: candidate n of channel c lives at c*N+n.
    // Each candidate gets a DISJOINT bbox so per-class NMS cannot merge them
    // (identical boxes would suppress down to the single best score).
    std::vector<float> flat(84 * 3, 0.0f);
    const std::vector<float> scores = {0.2f, 0.5f, 0.9f};
    for (size_t n = 0; n < scores.size(); ++n) {
        const auto candidate = one_candidate_yolov8(scores[n]);
        for (size_t c = 0; c < candidate.size(); ++c) {
            flat[c * scores.size() + n] = candidate[c];
        }
        // channel 0..3 = x, y, w, h: shift each candidate on the x axis
        flat[0 * scores.size() + n] = 10.0f + static_cast<float>(n) * 30.0f;
    }
    const auto tensor = output_tensor(DType::F32, {1, 84, 3}, flat);

    TestYoloV8Detector detector;
    detector._m_detection_params.score_threshold = 0.0f;
    detector._m_detection_params.nms_threshold = 0.9f;
    detector._m_detection_params.keep_top_k = 100;

    auto run_with_threshold = [&detector, &tensor](float request_threshold) {
        std_object_detection_output out;
        auto context = test_context();
        jinq::models::backend::ParamSet params;
        params.set_f32("score_threshold", request_threshold);
        context.params = &params;
        const auto status = detector.postprocess({tensor}, context, out);
        EXPECT_EQ(status, jinq::common::StatusCode::OK);
        return out.size();
    };

    // property-based sweep: a stricter threshold can never return MORE boxes
    EXPECT_EQ(run_with_threshold(0.0f), 3u);
    EXPECT_EQ(run_with_threshold(0.4f), 2u);
    EXPECT_EQ(run_with_threshold(0.6f), 1u);
    EXPECT_EQ(run_with_threshold(0.95f), 0u);

    // legacy path (nullptr params) keeps the pure config behavior
    {
        std_object_detection_output out;
        auto context = test_context();
        context.params = nullptr;
        detector._m_detection_params.score_threshold = 0.6f;
        EXPECT_EQ(detector.postprocess({tensor}, context, out), jinq::common::StatusCode::OK);
        EXPECT_EQ(out.size(), 1u);
    }
}
