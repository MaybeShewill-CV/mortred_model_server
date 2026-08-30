/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: response_serializers.h
* Date: 26-8-19
************************************************/

// Single source of truth for model-server response `data` payloads. Every
// concrete server delegates its fill_response_data() to one of these functions
// so field names and JSON types stay consistent. Schemas must match
// docs/openapi.json components.schemas.

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

/*** Unified bbox: [x1, y1, x2, y2] ***/
inline rapidjson::Value make_bbox(AllocatorType& allocator, const cv::Rect2f& bbox) {
    rapidjson::Value arr(rapidjson::kArrayType);
    arr.PushBack(bbox.x, allocator);
    arr.PushBack(bbox.y, allocator);
    arr.PushBack(bbox.x + bbox.width, allocator);
    arr.PushBack(bbox.y + bbox.height, allocator);
    return arr;
}

/*** Encode image as base64 string (empty image -> "") ***/
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

/*** Raw base64 image payload: image (already encoded by the model) ***/
inline void fill_base64_image(AllocatorType &allocator,
                              rapidjson::Document &data,
                              const jinq::models::io_define::common_io::base64_input &out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, out.input_image_content), allocator);
}

/*** Classification: class_id / category / scores ***/
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

/*** Object detection: class_id / score / category / bbox / detail_infos ***/
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

/*** Face detection: detection fields plus landmarks ***/
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

/*** OCR text regions: score / bbox / polygon / detail_infos ***/
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

/*** Scene segmentation: image + colorized_mask (PNG base64) ***/
inline void fill_scene_segmentation(
    AllocatorType& allocator,
    rapidjson::Document& data,
    const jinq::models::io_define::scene_segmentation::std_scene_segmentation_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.segmentation_result, ".png")),
                   allocator);
    cv::Mat color_mask = out.colorized_seg_mask;
    if (color_mask.empty() && !out.segmentation_result.empty()) {
        jinq::common::CvUtils::colorize_segmentation_mask(
            out.segmentation_result, color_mask, 80);
    }
    data.AddMember("colorized_mask",
                   make_string(allocator, encode_image(color_mask, ".png")), allocator);
}

/*** Matting: image (PNG base64) ***/
inline void fill_matting(AllocatorType& allocator,
                         rapidjson::Document& data,
                         const jinq::models::io_define::matting::std_matting_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.matting_result, ".png")),
                   allocator);
}

/*** Enhancement: image (JPG base64) ***/
inline void fill_enhancement(AllocatorType& allocator,
                             rapidjson::Document& data,
                             const jinq::models::io_define::enhancement::std_enhancement_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.enhancement_result, ".jpg")),
                   allocator);
}

/*** Mono depth: image (PNG base64, colorized depth map) ***/
inline void fill_depth_estimation(AllocatorType& allocator,
                                  rapidjson::Document& data,
                                  const jinq::models::io_define::mono_depth_estimation::std_mde_output& out) {
    data.SetObject();
    data.AddMember("image", make_string(allocator, encode_image(out.colorized_depth_map, ".png")),
                   allocator);
}

/*** Feature points: score / location / descriptor ***/
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

/*** SAM automatic masks: segmentation png + area / bbox / iou / stability ***/
inline void fill_sam_amg(AllocatorType &allocator,
                         rapidjson::Document &data,
                         const jinq::models::io_define::segment_anything::std_sam_amg_output &out) {
    const auto count = out.segmentations.size();
    data.SetArray();
    for (size_t index = 0; index < count; ++index) {
        rapidjson::Value item(rapidjson::kObjectType);
        item.AddMember("segmentation",
                       make_string(allocator, encode_image(out.segmentations[index], ".png")),
                       allocator);
        item.AddMember("area", index < out.areas.size() ? out.areas[index] : 0, allocator);
        if (index < out.bboxes.size()) {
            item.AddMember("bbox", make_bbox(allocator, out.bboxes[index]), allocator);
        }
        item.AddMember("predicted_iou",
                       index < out.preds_ious.size() ? out.preds_ious[index] : 0.0f,
                       allocator);
        item.AddMember("stability_score",
                       index < out.preds_stability_scores.size() ? out.preds_stability_scores[index] : 0.0f,
                       allocator);
        item.AddMember("detail_infos", rapidjson::Value(rapidjson::kObjectType), allocator);
        data.PushBack(item, allocator);
    }
}

}  // namespace response
}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_RESPONSE_SERIALIZERS_H
