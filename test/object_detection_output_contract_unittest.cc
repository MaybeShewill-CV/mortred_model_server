#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "models/backend/param_spec.h"

#include "models/backend/tensor.h"
#include "models/model_io_define.h"
#include "models/object_detection/yolov5_detector.h"
#include "models/object_detection/yolov7_detector.h"
#include "models/object_detection/yolov8_detector.h"
#include "rapidjson/document.h"
#include "server/http_status.h"
#include "server/item_exec.h"
#include "server/response_serializers.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::NamedTensor;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::object_detection::YoloV5Detector;
using jinq::models::object_detection::YoloV7Detector;
using jinq::models::object_detection::YoloV8Detector;

namespace {

class TestYoloV8Detector : public YoloV8Detector<mat_input, std_object_detection_output> {
  public:
    using YoloV8Detector<mat_input, std_object_detection_output>::postprocess;
    using YoloV8Detector<mat_input, std_object_detection_output>::_m_detection_params;
};

class TestYoloV5Detector : public YoloV5Detector<mat_input, std_object_detection_output> {
  public:
    using YoloV5Detector<mat_input, std_object_detection_output>::postprocess;
    using YoloV5Detector<mat_input, std_object_detection_output>::_m_detection_params;
};

class TestYoloV7Detector : public YoloV7Detector<mat_input, std_object_detection_output> {
  public:
    using YoloV7Detector<mat_input, std_object_detection_output>::postprocess;
    using YoloV7Detector<mat_input, std_object_detection_output>::_m_detection_params;
};

struct YoloV7Wire {
    int http = 0;
    int status = -1;
    std::string status_str;
    std::string model_name;
    bool partial = true;
    size_t result_count = 0;
    int item_status = -1;
    bool data_is_null = false;
    bool data_is_array = false;
};

jinq::models::backend::InferenceContext test_context();

NamedTensor yolov7_shaped_head(const std::string &name, int64_t height, int64_t width, int64_t channels) {
    NamedTensor output;
    output.name = name;
    output.tensor.dtype = DType::F32;
    output.tensor.shape = {1, 3, height, width, channels};
    const auto elements = static_cast<size_t>(1 * 3 * height * width * channels);
    output.tensor.buffer.assign(elements * sizeof(float), 0);
    return output;
}

YoloV7Wire yolov7_post_to_wire(const char *model_name, const std::vector<NamedTensor> &outputs) {
    TestYoloV7Detector detector;
    detector._m_detection_params.class_nums = 80;
    std_object_detection_output decoded;
    const StatusCode post_status = detector.postprocess(outputs, test_context(), decoded);

    jinq::server::InferenceResult<std_object_detection_output> result;
    result.item_status.assign(1, post_status);
    result.item_outputs.assign(1, std::move(decoded));
    jinq::server::aggregate_item_statuses(&result);

    const auto unified = jinq::server::inference_result_to_unified(
        "p0-3", model_name, result, &jinq::server::response::fill_object_detection);
    YoloV7Wire wire;
    wire.http = jinq::server::http_status_of(static_cast<StatusCode>(unified.status));
    const std::string body = jinq::common::envelope::encode(unified);

    rapidjson::Document doc;
    doc.Parse(body.data(), body.size());
    EXPECT_FALSE(doc.HasParseError());
    EXPECT_TRUE(doc.IsObject());
    if (!doc.IsObject()) {
        return wire;
    }
    if (doc.HasMember("status") && doc["status"].IsInt()) {
        wire.status = doc["status"].GetInt();
    }
    if (doc.HasMember("status_str") && doc["status_str"].IsString()) {
        wire.status_str.assign(doc["status_str"].GetString(), doc["status_str"].GetStringLength());
    }
    if (doc.HasMember("model") && doc["model"].IsObject()) {
        const auto &model = doc["model"];
        if (model.HasMember("name") && model["name"].IsString()) {
            wire.model_name.assign(model["name"].GetString(), model["name"].GetStringLength());
        }
    }
    if (doc.HasMember("partial") && doc["partial"].IsBool()) {
        wire.partial = doc["partial"].GetBool();
    }
    if (doc.HasMember("results") && doc["results"].IsArray()) {
        const auto &results = doc["results"];
        wire.result_count = results.Size();
        if (results.Size() == 1 && results[0].IsObject()) {
            if (results[0].HasMember("status") && results[0]["status"].IsInt()) {
                wire.item_status = results[0]["status"].GetInt();
            }
            if (results[0].HasMember("data")) {
                wire.data_is_null = results[0]["data"].IsNull();
                wire.data_is_array = results[0]["data"].IsArray();
            }
        }
    }
    EXPECT_EQ(wire.status, jinq::common::to_underlying(post_status));
    EXPECT_EQ(wire.status, unified.status);
    return wire;
}

void expect_yolov7_wire(const YoloV7Wire &wire, const char *model_name, int http, int status, const char *status_str,
                        bool data_null) {
    EXPECT_EQ(wire.http, http);
    EXPECT_EQ(wire.status, status);
    EXPECT_EQ(wire.status_str, status_str);
    EXPECT_EQ(wire.model_name, model_name);
    EXPECT_FALSE(wire.partial);
    EXPECT_EQ(wire.result_count, 1u);
    EXPECT_EQ(wire.item_status, status);
    if (data_null) {
        EXPECT_TRUE(wire.data_is_null);
    } else {
        EXPECT_FALSE(wire.data_is_null);
        EXPECT_TRUE(wire.data_is_array);
    }
}

bool yolov7_wire_matches(const YoloV7Wire &wire, int http, int status, const char *status_str) {
    return wire.http == http && wire.status == status && wire.status_str == status_str;
}

void print_yolov7_acceptance(int acceptance, const char *model_names, int http, int status, const char *status_str) {
    std::cout << "验收 " << acceptance << " " << model_names << " HTTP " << http << " status " << status << " " << status_str
              << std::endl;
}

NamedTensor output_tensor(DType dtype, const std::vector<int64_t> &shape, std::vector<float> values,
                          const std::string &name = "output0") {
    NamedTensor output;
    output.name = name;
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
    // source is 20x20; the unmapped box is clamped to that edge, so 10 becomes 5
    EXPECT_NEAR(result[0].bbox.width, 5.0f, 1e-5);
    EXPECT_NEAR(result[0].bbox.height, 5.0f, 1e-5);
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

TEST(ObjectDetectionOutputContract, YoloV8RejectsInfinity) {
    TestYoloV8Detector detector;
    detector._m_detection_params.class_nums = 80;
    detector._m_detection_params.score_threshold = 0.25f;
    std_object_detection_output result;
    const auto status = detector.postprocess(
        {output_tensor(DType::F32, {1, 84, 1}, one_candidate_yolov8(std::numeric_limits<float>::infinity()))},
        test_context(), result);
    EXPECT_EQ(status, StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);
    EXPECT_NE(status, StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_TRUE(result.empty());
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

TEST(ObjectDetectionOutputContract, YoloV8FiltersMinBoxAreaInNetworkPixels) {
    TestYoloV8Detector detector;
    detector._m_detection_params.class_nums = 80;
    detector._m_detection_params.score_threshold = 0.25f;
    detector._m_detection_params.min_box_area_px = 5.0f;

    std::vector<float> values(84, 0.0f);
    values[0] = 10.0f;
    values[1] = 10.0f;
    values[2] = 2.0f;
    values[3] = 2.0f;
    values[4 + 5] = 0.9f;
    std_object_detection_output result;
    ASSERT_EQ(detector.postprocess({output_tensor(DType::F32, {1, 84, 1}, values)}, test_context(), result), StatusCode::OK);
    EXPECT_TRUE(result.empty());
}

TEST(ObjectDetectionOutputContract, YoloV5FiltersMinBoxAreaInNetworkPixels) {
    TestYoloV5Detector detector;
    detector._m_detection_params.class_nums = 2;
    detector._m_detection_params.score_threshold = 0.25f;
    detector._m_detection_params.min_box_area_px = 5.0f;

    const std::vector<float> row = {10.0f, 10.0f, 2.0f, 2.0f, 1.0f, 0.9f, 0.1f};
    std_object_detection_output result;
    ASSERT_EQ(detector.postprocess({output_tensor(DType::F32, {1, 1, 7}, row, "output")}, test_context(), result), StatusCode::OK);
    EXPECT_TRUE(result.empty());
}

TEST(YoloV7OutputContract, Acceptance1NoMatchingHead) {
    const std::vector<NamedTensor> outputs = {output_tensor(DType::F32, {1, 84, 1}, std::vector<float>(84, 0.0f), "output0")};
    const auto wire = yolov7_post_to_wire("yolov7", outputs);
    expect_yolov7_wire(wire, "yolov7", 500, 6, "model output contract failed", true);
    if (yolov7_wire_matches(wire, 500, 6, "model output contract failed")) {
        print_yolov7_acceptance(1, "yolov7", 500, 6, "model output contract failed");
    }
}

TEST(YoloV7OutputContract, Acceptance2NoMatchingHead) {
    const std::vector<NamedTensor> outputs = {output_tensor(DType::F32, {1, 84, 1}, std::vector<float>(84, 0.0f), "output0")};
    const auto wire = yolov7_post_to_wire("yolov7x", outputs);
    expect_yolov7_wire(wire, "yolov7x", 500, 6, "model output contract failed", true);
    if (yolov7_wire_matches(wire, 500, 6, "model output contract failed")) {
        print_yolov7_acceptance(2, "yolov7x", 500, 6, "model output contract failed");
    }
}

TEST(YoloV7OutputContract, Acceptance3OneHead) {
    const std::vector<NamedTensor> outputs = {yolov7_shaped_head("h0", 8, 8, 1)};
    const auto yolov7 = yolov7_post_to_wire("yolov7", outputs);
    const auto yolov7x = yolov7_post_to_wire("yolov7x", outputs);
    expect_yolov7_wire(yolov7, "yolov7", 500, 6, "model output contract failed", true);
    expect_yolov7_wire(yolov7x, "yolov7x", 500, 6, "model output contract failed", true);
    if (yolov7_wire_matches(yolov7, 500, 6, "model output contract failed") &&
        yolov7_wire_matches(yolov7x, 500, 6, "model output contract failed")) {
        print_yolov7_acceptance(3, "yolov7 yolov7x", 500, 6, "model output contract failed");
    }
}

TEST(YoloV7OutputContract, Acceptance4TwoHeads) {
    const std::vector<NamedTensor> outputs = {yolov7_shaped_head("h0", 8, 8, 1), yolov7_shaped_head("h1", 4, 4, 1)};
    const auto yolov7 = yolov7_post_to_wire("yolov7", outputs);
    const auto yolov7x = yolov7_post_to_wire("yolov7x", outputs);
    expect_yolov7_wire(yolov7, "yolov7", 500, 6, "model output contract failed", true);
    expect_yolov7_wire(yolov7x, "yolov7x", 500, 6, "model output contract failed", true);
    if (yolov7_wire_matches(yolov7, 500, 6, "model output contract failed") &&
        yolov7_wire_matches(yolov7x, 500, 6, "model output contract failed")) {
        print_yolov7_acceptance(4, "yolov7 yolov7x", 500, 6, "model output contract failed");
    }
}

TEST(YoloV7OutputContract, Acceptance5MoreThanThreeHeads) {
    const std::vector<NamedTensor> outputs = {yolov7_shaped_head("h0", 8, 8, 1), yolov7_shaped_head("h1", 4, 4, 1),
                                              yolov7_shaped_head("h2", 2, 2, 1), yolov7_shaped_head("h3", 1, 1, 1)};
    const auto yolov7 = yolov7_post_to_wire("yolov7", outputs);
    const auto yolov7x = yolov7_post_to_wire("yolov7x", outputs);
    expect_yolov7_wire(yolov7, "yolov7", 500, 6, "model output contract failed", true);
    expect_yolov7_wire(yolov7x, "yolov7x", 500, 6, "model output contract failed", true);
    if (yolov7_wire_matches(yolov7, 500, 6, "model output contract failed") &&
        yolov7_wire_matches(yolov7x, 500, 6, "model output contract failed")) {
        print_yolov7_acceptance(5, "yolov7 yolov7x", 500, 6, "model output contract failed");
    }
}

TEST(YoloV7OutputContract, Acceptance6EmptyVector) {
    const std::vector<NamedTensor> outputs;
    const auto yolov7 = yolov7_post_to_wire("yolov7", outputs);
    const auto yolov7x = yolov7_post_to_wire("yolov7x", outputs);
    expect_yolov7_wire(yolov7, "yolov7", 500, 5, "model output empty", true);
    expect_yolov7_wire(yolov7x, "yolov7x", 500, 5, "model output empty", true);
    EXPECT_NE(yolov7.status, 6);
    EXPECT_NE(yolov7x.status, 6);
    if (yolov7_wire_matches(yolov7, 500, 5, "model output empty") && yolov7_wire_matches(yolov7x, 500, 5, "model output empty")) {
        print_yolov7_acceptance(6, "yolov7 yolov7x", 500, 5, "model output empty");
    }
}

TEST(YoloV7OutputContract, Acceptance7ThreeHeadsDecode) {
    const int channels = 80 + 5;
    const std::vector<NamedTensor> outputs = {yolov7_shaped_head("output", 4, 4, channels), yolov7_shaped_head("598", 2, 2, channels),
                                              yolov7_shaped_head("612", 1, 1, channels)};
    const auto yolov7 = yolov7_post_to_wire("yolov7", outputs);
    const auto yolov7x = yolov7_post_to_wire("yolov7x", outputs);
    expect_yolov7_wire(yolov7, "yolov7", 200, 0, "OK", false);
    expect_yolov7_wire(yolov7x, "yolov7x", 200, 0, "OK", false);
    if (yolov7_wire_matches(yolov7, 200, 0, "OK") && yolov7_wire_matches(yolov7x, 200, 0, "OK")) {
        print_yolov7_acceptance(7, "yolov7 yolov7x", 200, 0, "OK");
    }
}
