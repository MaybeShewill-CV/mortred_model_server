/************************************************
 * Author: Codex
 * File: response_serializers.h
 *
 * Single source of truth for model-server response `data` payloads.
 * Every concrete server delegates its fill_response_data() to one of these
 * functions, so field names and JSON types stay consistent across tasks.
 * The schemas here must match docs/openapi.json components.schemas.
 ************************************************/

#ifndef MORTRED_SERVER_RESPONSE_SERIALIZERS_H
#define MORTRED_SERVER_RESPONSE_SERIALIZERS_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "rapidjson/document.h"

#include "common/base64.h"
#include "common/cv_utils.h"
#include "models/model_io_define.h"

namespace jinq {
namespace server {
namespace response {

using AllocatorType = rapidjson::Document::AllocatorType;

/*** 统一 bbox 表示：[x1, y1, x2, y2] ***/
inline rapidjson::Value make_bbox(AllocatorType& allocator, const cv::Rect2f& bbox) {
    rapidjson::Value arr(rapidjson::kArrayType);
    arr.PushBack(bbox.x, allocator);
    arr.PushBack(bbox.y, allocator);
    arr.PushBack(bbox.x + bbox.width, allocator);
    arr.PushBack(bbox.y + bbox.height, allocator);
    return arr;
}

/*** 图像编码为 base64 字符串（空图返回空串） ***/
inline std::string encode_image(const cv::Mat& image, const char* ext) {
    if (image.empty()) {
        return "";
    }
    std::vector<uchar> buffer;
    cv::imencode(ext, image, buffer);
    return jinq::common::base64::encode(buffer.data(), buffer.size());
}

inline rapidjson::Value make_string(AllocatorType& allocator, const std::string& value) {
    return rapidjson::Value(value.c_str(), value.size(), allocator);
}

/*** 分类：class_id / category / scores ***/
inline void fill_classification(AllocatorType& allocator,
                                rapidjson::Document& data,
                                const jinq::models::io_define::classification::std_classification_output& out) {
    data.SetObject();
    data.AddMember("class_id", out.class_id, allocator);
    data.AddMember("category", make_string(allocator, out.category), allocator);
    rapidjson::Value scores(rapidjson::kArrayType);
    for (float s : out.scores) {
        scores.PushBack(s, allocator);
    }
    data.AddMember("scores", scores, allocator);
}

/*** 目标检测：class_id / score / category / bbox / detail_infos ***/
inline void fill_object_detection(AllocatorType& allocator,
                                  rapidjson::Document& data,
                                  const jinq::models::io_define::object_detection::std_object_detection_output& out) {
    data.SetArray();
    for (const auto& box : out) {
        rapidjson::Value item(rapidjson::kObjectType);
        item.AddMember("class_id", box.class_id, allocator);
        item.AddMember("score", box.score, allocator);
        item.AddMember("category", make_string(allocator, box.category), allocator);
        item.AddMember("bbox", make_bbox(allocator, box.bbox), allocator);
        item.AddMember("detail_infos", rapidjson::Value(rapidjson::kObjectType), allocator);
        data.PushBack(item, allocator);
    }
}

/*** 人脸检测：在检测基础上增加 landmarks ***/
inline void fill_face_detection(AllocatorType& allocator,
                                rapidjson::Document& data,
                                const jinq::models::io_define::object_detection::std_face_detection_output& out) {
    data.SetArray();
    for (const auto& box : out) {
        rapidjson::Value item(rapidjson::kObjectType);
        item.AddMember("class_id", box.class_id, allocator);
        item.AddMember("score", box.score, allocator);
        item.AddMember("category", make_string(allocator, box.category), allocator);
        item.AddMember("bbox", make_bbox(allocator, box.bbox), allocator);
        rapidjson::Value landmarks(rapidjson::kArrayType);
        for (const auto& pt : box.landmarks) {
            rapidjson::Value point(rapidjson::kArrayType);
            point.PushBack(pt.x, allocator);
            point.PushBack(pt.y, allocator);
            landmarks.PushBack(point, allocator);
        }
        item.AddMember("landmarks", landmarks, allocator);
        item.AddMember("detail_infos", rapidjson::Value(rapidjson::kObjectType), allocator);
        data.PushBack(item, allocator);
    }
}

/*** OCR 文本区域：score / bbox / polygon / detail_infos ***/
inline void fill_text_regions(AllocatorType& allocator,
                              rapidjson::Document& data,
                              const jinq::models::io_define::ocr::std_text_regions_output& out) {
    data.SetArray();
    for (const auto& region : out) {
        rapidjson::Value item(rapidjson::kObjectType);
        item.AddMember("score", region.score, allocator);
        item.AddMember("bbox", make_bbox(allocator, region.bbox), allocator);
        rapidjson::Value polygon(rapidjson::kArrayType);
        for (const auto& pt : region.polygon) {
            rapidjson::Value point(rapidjson::kArrayType);
            point.PushBack(pt.x, allocator);
            point.PushBack(pt.y, allocator);
            polygon.PushBack(point, allocator);
        }
        item.AddMember("polygon", polygon, allocator);
        item.AddMember("detail_infos", rapidjson::Value(rapidjson::kObjectType), allocator);
        data.PushBack(item, allocator);
    }
}

/*** 场景分割：image + colorized_mask（PNG base64） ***/
inline void fill_scene_segmentation(
    AllocatorType& allocator,
    rapidjson::Document& data,
    const jinq::models::io_define::scene_segmentation::std_scene_segmentation_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.segmentation_result, ".png")),
                   allocator);
    cv::Mat color_mask = out.colorized_seg_mask;
    if (color_mask.empty() && !out.segmentation_result.empty()) {
        jinq::common::cv_utils::colorize_segmentation_mask(
            out.segmentation_result, color_mask, 80);
    }
    data.AddMember("colorized_mask",
                   make_string(allocator, encode_image(color_mask, ".png")), allocator);
}

/*** 抠图：image（PNG base64） ***/
inline void fill_matting(AllocatorType& allocator,
                         rapidjson::Document& data,
                         const jinq::models::io_define::matting::std_matting_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.matting_result, ".png")),
                   allocator);
}

/*** 图像增强：image（JPG base64） ***/
inline void fill_enhancement(AllocatorType& allocator,
                             rapidjson::Document& data,
                             const jinq::models::io_define::enhancement::std_enhancement_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.enhancement_result, ".jpg")),
                   allocator);
}

/*** 单目深度：image（PNG base64，颜色化深度图） ***/
inline void fill_depth_estimation(AllocatorType& allocator,
                                  rapidjson::Document& data,
                                  const jinq::models::io_define::mono_depth_estimation::std_mde_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.colorized_depth_map, ".png")),
                   allocator);
}

/*** 特征点：score / location / descriptor ***/
inline void fill_feature_points(AllocatorType& allocator,
                                rapidjson::Document& data,
                                const jinq::models::io_define::feature_point::std_feature_point_output& out) {
    data.SetArray();
    for (const auto& fp : out) {
        rapidjson::Value item(rapidjson::kObjectType);
        item.AddMember("score", fp.score, allocator);
        rapidjson::Value location(rapidjson::kArrayType);
        location.PushBack(fp.location.x, allocator);
        location.PushBack(fp.location.y, allocator);
        item.AddMember("location", location, allocator);
        rapidjson::Value descriptor(rapidjson::kArrayType);
        for (float v : fp.descriptor) {
            descriptor.PushBack(v, allocator);
        }
        item.AddMember("descriptor", descriptor, allocator);
        data.PushBack(item, allocator);
    }
}

}  // namespace response
}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_RESPONSE_SERIALIZERS_H
