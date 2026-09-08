#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "models/model_io_define.h"
#include "models/object_detection/detector_common.h"
#include "models/object_detection/yolo_cxcywh_decode.h"
#include "models/object_detection/yolov7_decode.h"
#include "models/object_detection/yolov8_decode.h"

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::GeometryScale;
using jinq::models::backend::LetterboxGeometry;
using jinq::models::backend::compute_letterbox_geometry;
using jinq::models::backend::make_geometry_scale;
using jinq::models::backend::make_letterbox_geometry;
using jinq::models::backend::scale_bbox;
using jinq::models::backend::scale_point;
using jinq::models::backend::unmap_letterbox_bbox;
using jinq::models::backend::validated_f32_named_output;
using jinq::models::object_detection::DetectionParams;
using jinq::models::object_detection::F32OutputView;
using jinq::models::object_detection::collect_yolo_cxcywh_obj_candidates;
using jinq::models::object_detection::collect_yolov7_head_candidates;
using jinq::models::object_detection::collect_yolov8_candidates;
using jinq::models::object_detection::make_nchw_input;
using jinq::models::object_detection::passes_min_box_area;
using jinq::models::object_detection::unmap_letterbox_detections;

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
    GeometryScale scale;
    std::string error;
    ASSERT_TRUE(make_geometry_scale(context, &scale, &error)) << error;
    EXPECT_FLOAT_EQ(scale.width, 2.0f);
    EXPECT_FLOAT_EQ(scale.height, 6.0f);

    const auto bbox = scale_bbox({1.0f, 2.0f, 3.0f, 4.0f}, scale);
    EXPECT_FLOAT_EQ(bbox.x, 2.0f);
    EXPECT_FLOAT_EQ(bbox.y, 12.0f);
    EXPECT_FLOAT_EQ(bbox.width, 6.0f);
    EXPECT_FLOAT_EQ(bbox.height, 24.0f);

    const auto point = scale_point({1.5f, 2.5f}, scale);
    EXPECT_FLOAT_EQ(point.x, 3.0f);
    EXPECT_FLOAT_EQ(point.y, 15.0f);
}

TEST(DetectorCommon, RejectsInvalidRequestGeometry) {
    auto context = test_context();
    context.network_size = cv::Size();
    GeometryScale scale;
    std::string error;
    EXPECT_FALSE(make_geometry_scale(context, &scale, &error));
    EXPECT_NE(error.find("invalid request geometry"), std::string::npos);
}

TEST(DetectorCommon, ValidatesNamedF32Output) {
    std::vector<NamedTensor> outputs{valid_output()};
    F32OutputView view;
    const auto status = validated_f32_named_output(outputs, "output", {DType::F32, 3, {1, 84, 1}}, "test detector", &view);
    ASSERT_EQ(status, StatusCode::OK);
    ASSERT_NE(view.tensor, nullptr);
    ASSERT_NE(view.data, nullptr);
    EXPECT_EQ(view.tensor->shape, std::vector<int64_t>({1, 84, 1}));
    EXPECT_EQ(view.data[0], 0.0f);
}

TEST(DetectorCommon, DistinguishesMissingAndContractFailedOutputs) {
    std::vector<NamedTensor> outputs{valid_output()};
    EXPECT_EQ(validated_f32_named_output(outputs, "missing", {DType::F32, 1, {1}}, "test detector"), StatusCode::MODEL_EMPTY_OUTPUT);
    EXPECT_EQ(validated_f32_named_output(outputs, "output", {DType::F32, 3, {1, 83, 1}}, "test detector"),
              StatusCode::MODEL_OUTPUT_CONTRACT_FAILED);

    auto non_finite = valid_output();
    *reinterpret_cast<float *>(non_finite.tensor.buffer.data()) = std::numeric_limits<float>::quiet_NaN();
    outputs[0] = non_finite;
    EXPECT_EQ(validated_f32_named_output(outputs, "output", {DType::F32, 3, {1, 84, 1}}, "test detector"),
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

TEST(DetectorCommon, RequestParamsOverrideConfigDefaults) {
    using jinq::models::io_define::object_detection::bbox;

    DetectionParams params;
    params.class_nums = 1;
    params.class_names = {"person"};
    params.score_threshold = 0.1f;
    params.keep_top_k = 10;

    std::vector<bbox> detections;
    for (int idx = 0; idx < 3; ++idx) {
        bbox detection;
        detection.bbox = cv::Rect2f(idx * 20.0f, 0.0f, 10.0f, 10.0f);
        detection.score = 0.2f + 0.2f * idx;  // 0.2 / 0.4 / 0.6
        detection.class_id = 0;
        detections.push_back(detection);
    }

    // legacy path: nullptr params keeps pure config behavior
    auto context = test_context();
    context.params = nullptr;
    const auto legacy = jinq::models::object_detection::finalize_detections(detections, params, context);
    EXPECT_EQ(legacy.size(), 3u);

    // request override: score_threshold 0.5 keeps only the 0.6 box,
    // top_k 1 would not change that; check threshold first
    jinq::models::backend::ParamSet request_params;
    request_params.set_f32("score_threshold", 0.5f);
    context.params = &request_params;
    const auto strict = jinq::models::object_detection::finalize_detections(detections, params, context);
    ASSERT_EQ(strict.size(), 1u);
    EXPECT_FLOAT_EQ(strict[0].score, 0.6f);

    // top_k truncation also overrides the config default
    request_params = jinq::models::backend::ParamSet();
    request_params.set_f32("score_threshold", 0.1f);
    request_params.set_i32("top_k", 2);
    const auto truncated = jinq::models::object_detection::finalize_detections(detections, params, context);
    EXPECT_EQ(truncated.size(), 2u);
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

TEST(DetectorCommon, ComputesUltralyticsLetterboxAndUnmapsBoxes) {
    jinq::models::backend::InferenceContext context;
    context.source_size = cv::Size(800, 600);
    context.network_size = cv::Size(640, 640);

    LetterboxGeometry geom;
    std::string error;
    ASSERT_TRUE(make_letterbox_geometry(context, &geom, &error)) << error;
    EXPECT_FLOAT_EQ(geom.scale, 0.8f);
    EXPECT_EQ(geom.unpadded, cv::Size(640, 480));
    EXPECT_EQ(geom.pad_x, 0);
    EXPECT_EQ(geom.pad_y, 80);
    EXPECT_EQ(geom.pad_right, 0);
    EXPECT_EQ(geom.pad_bottom, 80);

    const auto identity = compute_letterbox_geometry({640, 640}, {640, 640});
    EXPECT_FLOAT_EQ(identity.scale, 1.0f);
    EXPECT_EQ(identity.pad_x, 0);
    EXPECT_EQ(identity.pad_y, 0);

    const auto mapped = unmap_letterbox_bbox({80.0f, 120.0f, 64.0f, 32.0f}, geom, context.source_size);
    EXPECT_FLOAT_EQ(mapped.x, 100.0f);
    EXPECT_FLOAT_EQ(mapped.y, 50.0f);
    EXPECT_FLOAT_EQ(mapped.width, 80.0f);
    EXPECT_FLOAT_EQ(mapped.height, 40.0f);

    const auto clipped = unmap_letterbox_bbox({0.0f, 0.0f, 10.0f, 40.0f}, geom, context.source_size);
    EXPECT_FLOAT_EQ(clipped.x, 0.0f);
    EXPECT_FLOAT_EQ(clipped.y, 0.0f);
    EXPECT_FLOAT_EQ(clipped.width, 12.5f);
    EXPECT_FLOAT_EQ(clipped.height, 0.0f);

    context.network_size = cv::Size();
    EXPECT_FALSE(make_letterbox_geometry(context, &geom, &error));
    EXPECT_NE(error.find("invalid request geometry"), std::string::npos);
}

TEST(DetectorCommon, YoloV8SyntheticDecodeNmsThenLetterboxUnmap) {
    constexpr int64_t k_rows = 6;
    constexpr int64_t k_proposals = 3;
    std::vector<float> packed(static_cast<size_t>(k_rows * k_proposals), 0.0f);
    packed[0 * k_proposals + 0] = 320.0f;
    packed[0 * k_proposals + 1] = 322.0f;
    packed[0 * k_proposals + 2] = 100.0f;
    packed[1 * k_proposals + 0] = 320.0f;
    packed[1 * k_proposals + 1] = 322.0f;
    packed[1 * k_proposals + 2] = 100.0f;
    packed[2 * k_proposals + 0] = 100.0f;
    packed[2 * k_proposals + 1] = 100.0f;
    packed[2 * k_proposals + 2] = 40.0f;
    packed[3 * k_proposals + 0] = 80.0f;
    packed[3 * k_proposals + 1] = 80.0f;
    packed[3 * k_proposals + 2] = 40.0f;
    packed[4 * k_proposals + 0] = 0.9f;
    packed[4 * k_proposals + 1] = 0.8f;
    packed[4 * k_proposals + 2] = 0.1f;
    packed[5 * k_proposals + 0] = 0.1f;
    packed[5 * k_proposals + 1] = 0.1f;
    packed[5 * k_proposals + 2] = 0.7f;

    jinq::models::io_define::object_detection::std_object_detection_output candidates;
    collect_yolov8_candidates(packed.data(), k_rows, k_proposals, 0.5f, 5.0f, &candidates);
    ASSERT_EQ(candidates.size(), 3u);

    DetectionParams params;
    params.class_nums = 2;
    params.class_names = {"person", "bicycle"};
    params.score_threshold = 0.5f;
    params.nms_threshold = 0.5f;
    params.keep_top_k = 10;

    auto kept = jinq::models::object_detection::finalize_detections(std::move(candidates), params);
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[0].class_id, 0);
    EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
    EXPECT_EQ(kept[0].category, "person");
    EXPECT_EQ(kept[1].class_id, 1);
    EXPECT_FLOAT_EQ(kept[1].score, 0.7f);
    EXPECT_EQ(kept[1].category, "bicycle");

    jinq::models::backend::InferenceContext context;
    context.source_size = cv::Size(800, 600);
    context.network_size = cv::Size(640, 640);
    LetterboxGeometry geom;
    std::string error;
    ASSERT_TRUE(make_letterbox_geometry(context, &geom, &error)) << error;
    unmap_letterbox_detections(kept, geom, context.source_size);

    EXPECT_FLOAT_EQ(kept[0].bbox.x, 337.5f);
    EXPECT_FLOAT_EQ(kept[0].bbox.y, 250.0f);
    EXPECT_FLOAT_EQ(kept[0].bbox.width, 125.0f);
    EXPECT_FLOAT_EQ(kept[0].bbox.height, 100.0f);
    EXPECT_FLOAT_EQ(kept[1].bbox.x, 100.0f);
    EXPECT_FLOAT_EQ(kept[1].bbox.y, 0.0f);
    EXPECT_FLOAT_EQ(kept[1].bbox.width, 50.0f);
    EXPECT_FLOAT_EQ(kept[1].bbox.height, 50.0f);
}

TEST(DetectorCommon, YoloCxcywhObjSyntheticDecodeFiltersNetworkArea) {
    // Two packed [1, 2, 7] rows (v5/v6): 2x2 area=4 must drop at min=5;
    // 3x3 area=9 stays. Boxes remain in network pixels.
    constexpr int class_nums = 2;
    constexpr int64_t proposals = 2;
    const float small_row[] = {10.0f, 10.0f, 2.0f, 2.0f, 1.0f, 0.9f, 0.1f};
    const float large_row[] = {30.0f, 10.0f, 3.0f, 3.0f, 1.0f, 0.9f, 0.1f};
    std::vector<float> packed;
    packed.insert(packed.end(), small_row, small_row + 7);
    packed.insert(packed.end(), large_row, large_row + 7);

    jinq::models::io_define::object_detection::std_object_detection_output candidates;
    collect_yolo_cxcywh_obj_candidates(packed.data(), 1, proposals, class_nums, 0.5f, 5.0f, &candidates);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_FLOAT_EQ(candidates[0].bbox.width, 3.0f);
    EXPECT_FLOAT_EQ(candidates[0].bbox.height, 3.0f);

    candidates.clear();
    collect_yolo_cxcywh_obj_candidates(packed.data(), 1, proposals, class_nums, 0.5f, 0.0f, &candidates);
    EXPECT_EQ(candidates.size(), 2u);

    // Contrast: the dropped 2x2 network box unmaps to 4x4=16 source px when
    // letterbox scale is 0.5, which would have passed a post-unmap filter.
    jinq::models::backend::InferenceContext context;
    context.source_size = cv::Size(400, 400);
    context.network_size = cv::Size(200, 200);
    LetterboxGeometry geom;
    std::string error;
    ASSERT_TRUE(make_letterbox_geometry(context, &geom, &error)) << error;
    EXPECT_FLOAT_EQ(geom.scale, 0.5f);
    const auto unmapped_small = unmap_letterbox_bbox({9.0f, 9.0f, 2.0f, 2.0f}, geom, context.source_size);
    EXPECT_FLOAT_EQ(unmapped_small.area(), 16.0f);
    EXPECT_TRUE(passes_min_box_area(unmapped_small, 5.0f));
}

TEST(DetectorCommon, YoloV7SyntheticDecodeFiltersNetworkArea) {
    // 1x1 grid, 1 class (attrs=6). Anchor 0 with dw/dh logits -2.2 decodes to
    // a sub-5px network box; anchor 1 with zeros decodes to the 19x36 anchor.
    constexpr int attrs = 6;
    constexpr int grid = 1;
    constexpr int anchor_nums = 3;
    std::vector<float> data(static_cast<size_t>(anchor_nums * grid * grid * attrs), -10.0f);
    const float anchors[3][2] = {{12.0f, 16.0f}, {19.0f, 36.0f}, {40.0f, 28.0f}};
    auto fill = [&](int anchor, float dx, float dy, float dw, float dh, float obj, float cls) {
        float *p = data.data() + anchor * attrs;
        p[0] = dx;
        p[1] = dy;
        p[2] = dw;
        p[3] = dh;
        p[4] = obj;
        p[5] = cls;
    };
    fill(0, 0.0f, 0.0f, -2.2f, -2.2f, 6.0f, 6.0f);
    fill(1, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 6.0f);

    jinq::models::io_define::object_detection::std_object_detection_output candidates;
    collect_yolov7_head_candidates(data.data(), anchor_nums, grid, grid, attrs, 8, anchors, 0.4f, 5.0f, &candidates);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_NEAR(candidates[0].bbox.width, 19.0f, 1e-4);
    EXPECT_NEAR(candidates[0].bbox.height, 36.0f, 1e-4);
}

TEST(DetectorCommon, YoloV8SyntheticDecodeFiltersNetworkArea) {
    constexpr int64_t k_rows = 6;
    constexpr int64_t k_proposals = 2;
    std::vector<float> packed(static_cast<size_t>(k_rows * k_proposals), 0.0f);
    packed[0 * k_proposals + 0] = 10.0f;
    packed[0 * k_proposals + 1] = 30.0f;
    packed[1 * k_proposals + 0] = 10.0f;
    packed[1 * k_proposals + 1] = 10.0f;
    packed[2 * k_proposals + 0] = 2.0f;
    packed[2 * k_proposals + 1] = 3.0f;
    packed[3 * k_proposals + 0] = 2.0f;
    packed[3 * k_proposals + 1] = 3.0f;
    packed[4 * k_proposals + 0] = 0.9f;
    packed[4 * k_proposals + 1] = 0.9f;
    packed[5 * k_proposals + 0] = 0.1f;
    packed[5 * k_proposals + 1] = 0.1f;

    jinq::models::io_define::object_detection::std_object_detection_output candidates;
    collect_yolov8_candidates(packed.data(), k_rows, k_proposals, 0.5f, 5.0f, &candidates);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_FLOAT_EQ(candidates[0].bbox.width, 3.0f);
    EXPECT_FLOAT_EQ(candidates[0].bbox.height, 3.0f);
}
