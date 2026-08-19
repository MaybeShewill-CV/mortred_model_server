/************************************************
 * Author: Codex
 * File: response_schema_test.cc
 *
 * 校验 src/server/response_serializers.h 的输出 schema：
 * 键名与 JSON 类型必须与 docs/openapi.json components.schemas 一致。
 * 这类测试防止"数字被序列化成字符串/字段改名"类回归。
 ************************************************/

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>

#include "server/response_serializers.h"

using jinq::server::response::fill_classification;
using jinq::server::response::fill_depth_estimation;
using jinq::server::response::fill_enhancement;
using jinq::server::response::fill_face_detection;
using jinq::server::response::fill_feature_points;
using jinq::server::response::fill_matting;
using jinq::server::response::fill_object_detection;
using jinq::server::response::fill_scene_segmentation;
using jinq::server::response::fill_text_regions;
using jinq::models::io_define::classification::std_classification_output;
using jinq::models::io_define::enhancement::std_enhancement_output;
using jinq::models::io_define::feature_point::std_feature_point_output;
using jinq::models::io_define::matting::std_matting_output;
using jinq::models::io_define::mono_depth_estimation::std_mde_output;
using jinq::models::io_define::object_detection::bbox;
using jinq::models::io_define::object_detection::face_bbox;
using jinq::models::io_define::object_detection::std_face_detection_output;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::io_define::ocr::std_text_regions_output;
using jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;

namespace {

using AllocatorType = rapidjson::Document::AllocatorType;

template<typename F, typename T>
rapidjson::Document serialize(F filler, const T& output) {
    rapidjson::Document data;
    filler(data.GetAllocator(), data, output);
    return data;
}

void expect_only_members(const rapidjson::Value& obj,
                         const std::vector<std::string>& expected) {
    ASSERT_TRUE(obj.IsObject());
    EXPECT_EQ(obj.MemberCount(), expected.size());
    for (const auto& key : expected) {
        EXPECT_TRUE(obj.HasMember(key.c_str())) << "missing member: " << key;
    }
}

cv::Mat make_small_image() {
    return cv::Mat(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
}

}  // namespace

TEST(response_schema, classification_uses_numeric_class_id_and_scores) {
    std_classification_output out;
    out.class_id = 3;
    out.category = "tabby cat";
    out.scores = {0.1f, 0.8f, 0.1f};

    auto data = serialize([](AllocatorType& a, rapidjson::Document& d, const std_classification_output& o) {
        fill_classification(a, d, o);
    }, out);

    expect_only_members(data, {"class_id", "category", "scores"});
    EXPECT_TRUE(data["class_id"].IsInt());
    EXPECT_TRUE(data["category"].IsString());
    EXPECT_TRUE(data["scores"].IsArray());
    for (const auto& s : data["scores"].GetArray()) {
        EXPECT_TRUE(s.IsNumber());
    }
    EXPECT_FALSE(data.HasMember("cls_id"));
}

TEST(response_schema, detection_uses_class_id_bbox_and_numeric_score) {
    std_object_detection_output out;
    bbox box;
    box.bbox = cv::Rect2f(1.0f, 2.0f, 10.0f, 20.0f);
    box.score = 0.93f;
    box.class_id = 2;
    box.category = "cat";
    out.push_back(box);

    auto data = serialize([](AllocatorType& a, rapidjson::Document& d, const std_object_detection_output& o) {
        fill_object_detection(a, d, o);
    }, out);

    ASSERT_TRUE(data.IsArray());
    ASSERT_EQ(data.Size(), 1u);
    expect_only_members(data[0], {"class_id", "score", "category", "bbox", "detail_infos"});
    EXPECT_TRUE(data[0]["class_id"].IsInt());
    EXPECT_TRUE(data[0]["score"].IsDouble());
    EXPECT_TRUE(data[0]["category"].IsString());
    ASSERT_TRUE(data[0]["bbox"].IsArray());
    EXPECT_EQ(data[0]["bbox"].Size(), 4u);
    for (const auto& v : data[0]["bbox"].GetArray()) {
        EXPECT_TRUE(v.IsNumber());
    }
    EXPECT_FALSE(data[0].HasMember("cls_id"));
    EXPECT_FALSE(data[0].HasMember("points"));
}

TEST(response_schema, face_detection_uses_landmarks_plural) {
    std_face_detection_output out;
    face_bbox box;
    box.bbox = cv::Rect2f(0, 0, 10, 10);
    box.score = 0.9f;
    box.class_id = 0;
    box.category = "face";
    box.landmarks = {cv::Point2f(1, 1), cv::Point2f(2, 2)};
    out.push_back(box);

    auto data = serialize([](AllocatorType& a, rapidjson::Document& d, const std_face_detection_output& o) {
        fill_face_detection(a, d, o);
    }, out);

    ASSERT_TRUE(data.IsArray());
    expect_only_members(data[0],
                        {"class_id", "score", "category", "bbox", "landmarks", "detail_infos"});
    ASSERT_TRUE(data[0]["landmarks"].IsArray());
    EXPECT_EQ(data[0]["landmarks"].Size(), 2u);
    EXPECT_FALSE(data[0].HasMember("landmark"));
    EXPECT_FALSE(data[0].HasMember("box"));
}

TEST(response_schema, ocr_uses_bbox_and_polygon) {
    std_text_regions_output out;
    jinq::models::io_define::ocr::text_region region;
    region.bbox = cv::Rect2f(0, 0, 5, 5);
    region.score = 0.8f;
    region.polygon = {cv::Point2f(0, 0), cv::Point2f(5, 0), cv::Point2f(5, 5), cv::Point2f(0, 5)};
    out.push_back(region);

    auto data = serialize([](AllocatorType& a, rapidjson::Document& d, const std_text_regions_output& o) {
        fill_text_regions(a, d, o);
    }, out);

    ASSERT_TRUE(data.IsArray());
    expect_only_members(data[0], {"score", "bbox", "polygon", "detail_infos"});
    EXPECT_TRUE(data[0]["score"].IsDouble());
    ASSERT_TRUE(data[0]["polygon"].IsArray());
    EXPECT_EQ(data[0]["polygon"].Size(), 4u);
}

TEST(response_schema, image_outputs_use_image_key) {
    std_scene_segmentation_output seg;
    auto seg_data = serialize(
        [](AllocatorType& a, rapidjson::Document& d, const std_scene_segmentation_output& o) {
            fill_scene_segmentation(a, d, o);
        },
        seg);
    expect_only_members(seg_data, {"image", "colorized_mask"});
    EXPECT_TRUE(seg_data["image"].IsString());
    EXPECT_TRUE(seg_data["colorized_mask"].IsString());
    EXPECT_FALSE(seg_data.HasMember("segment_result"));
    EXPECT_FALSE(seg_data.HasMember("colorized_seg_mask"));

    std_matting_output matting;
    auto matting_data = serialize(
        [](AllocatorType& a, rapidjson::Document& d, const std_matting_output& o) {
            fill_matting(a, d, o);
        },
        matting);
    expect_only_members(matting_data, {"image"});

    std_enhancement_output enhance;
    enhance.enhancement_result = make_small_image();
    auto enhance_data = serialize(
        [](AllocatorType& a, rapidjson::Document& d, const std_enhancement_output& o) {
            fill_enhancement(a, d, o);
        },
        enhance);
    expect_only_members(enhance_data, {"image"});
    EXPECT_FALSE(enhance_data["image"].GetStringLength() == 0);
    EXPECT_FALSE(enhance_data.HasMember("enhance_result"));

    std_mde_output depth;
    auto depth_data = serialize(
        [](AllocatorType& a, rapidjson::Document& d, const std_mde_output& o) {
            fill_depth_estimation(a, d, o);
        },
        depth);
    expect_only_members(depth_data, {"image"});
    EXPECT_FALSE(depth_data.HasMember("estimate_result"));
}

TEST(response_schema, feature_points_use_location_and_descriptor) {
    std_feature_point_output out;
    jinq::models::io_define::feature_point::fp point;
    point.location = cv::Point2f(3.5f, 4.5f);
    point.score = 0.7f;
    point.descriptor = {0.1f, 0.2f, 0.3f};
    out.push_back(point);

    auto data = serialize([](AllocatorType& a, rapidjson::Document& d, const std_feature_point_output& o) {
        fill_feature_points(a, d, o);
    }, out);

    ASSERT_TRUE(data.IsArray());
    expect_only_members(data[0], {"score", "location", "descriptor"});
    EXPECT_TRUE(data[0]["score"].IsDouble());
    ASSERT_TRUE(data[0]["location"].IsArray());
    EXPECT_EQ(data[0]["location"].Size(), 2u);
    ASSERT_TRUE(data[0]["descriptor"].IsArray());
    EXPECT_EQ(data[0]["descriptor"].Size(), 3u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
