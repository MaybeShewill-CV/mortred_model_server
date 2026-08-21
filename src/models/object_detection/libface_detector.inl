/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.inl
 * Date: 22-6-10
 ************************************************/

#include "libface_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace object_detection {

using FaceBBox = jinq::models::io_define::object_detection::face_bbox;
using FaceOutput = jinq::models::io_define::object_detection::std_face_detection_output;
using jinq::models::backend::NamedTensor;
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
StatusCode LibFaceDetector<INPUT, OUTPUT>::on_init(const toml::table& params) {
    if (params.contains("model_score_threshold")) {
        _m_score_threshold = params["model_score_threshold"].value_or<double>(0.0);
    }
    _m_score_threshold = std::max(_m_score_threshold, 0.5);
    if (params.contains("model_nms_threshold")) {
        _m_nms_threshold = params["model_nms_threshold"].value_or<double>(0.0);
    }
    if (params.contains("model_keep_top_k")) {
        _m_keep_topk = static_cast<size_t>(params["model_keep_top_k"].value_or<int64_t>(0));
    }

    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected libface input shape: " << input_info.to_string()
                   << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
const NamedTensor* LibFaceDetector<INPUT, OUTPUT>::find_output(
    const std::vector<NamedTensor>& outputs, const std::string& name) const {
    for (const auto& item : outputs) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

template<typename INPUT, typename OUTPUT>
auto LibFaceDetector<INPUT, OUTPUT>::generate_prior_anchors() const
    -> std::vector<FaceAnchor> {
    const std::vector<std::vector<double>> min_sizes = {
        {10., 16., 24.}, {32., 48.}, {64., 96.}, {128., 192., 256.}};
    const std::vector<double> steps = {8., 16., 32., 64.};

    const auto in_h = _m_input_size_host.height;
    const auto in_w = _m_input_size_host.width;
    const std::vector<int> feature_map_2th = {
        static_cast<int>((in_h + 1) / 2 / 2), static_cast<int>((in_w + 1) / 2 / 2)};
    const std::vector<int> feature_map_3th = {feature_map_2th[0] / 2, feature_map_2th[1] / 2};
    const std::vector<int> feature_map_4th = {feature_map_3th[0] / 2, feature_map_3th[1] / 2};
    const std::vector<int> feature_map_5th = {feature_map_4th[0] / 2, feature_map_4th[1] / 2};
    const std::vector<int> feature_map_6th = {feature_map_5th[0] / 2, feature_map_5th[1] / 2};
    const std::vector<std::vector<int>> feature_maps = {
        feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th};

    std::vector<FaceAnchor> anchors;
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        const auto& feature_map = feature_maps[k];
        const auto& feature_min_sizes = min_sizes[k];
        for (int i = 0; i < feature_map[0]; ++i) {
            for (int j = 0; j < feature_map[1]; ++j) {
                for (const auto min_size : feature_min_sizes) {
                    FaceAnchor anchor;
                    anchor.s_kx = min_size / in_w;
                    anchor.s_ky = min_size / in_h;
                    anchor.cx = (static_cast<double>(j) + 0.5) * steps[k] / in_w;
                    anchor.cy = (static_cast<double>(i) + 0.5) * steps[k] / in_h;
                    anchors.push_back(anchor);
                }
            }
        }
    }
    return anchors;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> LibFaceDetector<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    _m_input_size_user = input_image.size();
    cv::Mat tmp;
    if (input_image.size() != _m_input_size_host) {
        cv::resize(input_image, tmp, _m_input_size_host);
    } else {
        input_image.copyTo(tmp);
    }
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed libface image mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
StatusCode LibFaceDetector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                        OUTPUT& output) {
    const auto* loc = find_output(outputs, "loc");
    const auto* conf = find_output(outputs, "conf");
    if (loc == nullptr || conf == nullptr) {
        LOG(ERROR) << "libface outputs loc/conf missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    if (loc->tensor.dtype != jinq::models::backend::DType::F32 ||
        conf->tensor.dtype != jinq::models::backend::DType::F32) {
        LOG(ERROR) << "libface outputs must be f32 tensors";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    if (loc->tensor.shape.size() != 3 || conf->tensor.shape.size() != 3 ||
        loc->tensor.shape[0] != 1 || conf->tensor.shape[0] != 1 ||
        loc->tensor.shape[1] != conf->tensor.shape[1] || loc->tensor.shape[2] != 14 ||
        conf->tensor.shape[2] != 2) {
        LOG(ERROR) << "unexpected libface output shapes: loc="
                   << jinq::models::backend::shape_to_string(loc->tensor.shape) << ", conf="
                   << jinq::models::backend::shape_to_string(conf->tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto anchor_count = static_cast<size_t>(loc->tensor.shape[1]);
    const auto* loc_data = loc->tensor.template data<float>();
    const auto* conf_data = conf->tensor.template data<float>();

    // MNN rank-3 host output is row-major [N, anchors, channels].  A real
    // 640x480-weight probe found 37 threshold hits with this interleaved view
    // versus 8776 with the old channel-major view; only the former matches the
    // five-box golden result after anchor decoding and NMS.

    const auto priors = generate_prior_anchors();
    if (priors.size() != anchor_count) {
        LOG(ERROR) << "libface anchor count " << priors.size()
                   << " mismatches output anchor count " << anchor_count;
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    std::vector<FaceBBox> decode_result;
    for (size_t bbox_index = 0; bbox_index < anchor_count; ++bbox_index) {
        const auto& prior = priors[bbox_index];
        const auto raw_conf = conf_data[bbox_index * 2 + 1];
        if (raw_conf <= _m_score_threshold) {
            continue;
        }

        const auto raw_bbox_x = loc_data[bbox_index * 14];
        const auto raw_bbox_y = loc_data[bbox_index * 14 + 1];
        const auto raw_bbox_w = loc_data[bbox_index * 14 + 2];
        const auto raw_bbox_h = loc_data[bbox_index * 14 + 3];
        auto pred_bbox_x = prior.cx + raw_bbox_x * 0.1 * prior.s_kx;
        auto pred_bbox_y = prior.cy + raw_bbox_y * 0.1 * prior.s_ky;
        auto pred_bbox_w = prior.s_kx * std::exp(raw_bbox_w * 0.2);
        auto pred_bbox_h = prior.s_ky * std::exp(raw_bbox_h * 0.2);
        pred_bbox_x = (pred_bbox_x - pred_bbox_w / 2.0) * _m_input_size_host.width;
        pred_bbox_y = (pred_bbox_y - pred_bbox_h / 2.0) * _m_input_size_host.height;
        pred_bbox_w *= _m_input_size_host.width;
        pred_bbox_h *= _m_input_size_host.height;

        FaceBBox face_box;
        face_box.score = raw_conf;
        face_box.bbox = cv::Rect2f(
            static_cast<float>(pred_bbox_x), static_cast<float>(pred_bbox_y),
            static_cast<float>(pred_bbox_w), static_cast<float>(pred_bbox_h));
        for (size_t landmark_index = 4; landmark_index < 14; landmark_index += 2) {
            const auto raw_landmark_x = loc_data[bbox_index * 14 + landmark_index];
            const auto raw_landmark_y = loc_data[bbox_index * 14 + landmark_index + 1];
            const auto pred_landmark_x =
                (prior.cx + raw_landmark_x * 0.1 * prior.s_kx) * _m_input_size_host.width;
            const auto pred_landmark_y =
                (prior.cy + raw_landmark_y * 0.1 * prior.s_ky) * _m_input_size_host.height;
            face_box.landmarks.emplace_back(
                static_cast<float>(pred_landmark_x), static_cast<float>(pred_landmark_y));
        }
        face_box.class_id = 0;
        decode_result.push_back(std::move(face_box));
    }

    auto nms_result = jinq::common::CvUtils::nms_bboxes(decode_result, _m_nms_threshold);
    if (nms_result.size() > _m_keep_topk) {
        nms_result.resize(_m_keep_topk);
    }

    const auto width_scale =
        _m_input_size_user.width / static_cast<float>(_m_input_size_host.width);
    const auto height_scale =
        _m_input_size_user.height / static_cast<float>(_m_input_size_host.height);
    for (auto& face_box : nms_result) {
        face_box.bbox.x *= width_scale;
        face_box.bbox.y *= height_scale;
        face_box.bbox.width *= width_scale;
        face_box.bbox.height *= height_scale;
        for (auto& landmark : face_box.landmarks) {
            landmark.x *= width_scale;
            landmark.y *= height_scale;
        }
        face_box.category = "face";
    }

    FaceOutput faces;
    faces.reserve(nms_result.size());
    for (const auto& face_box : nms_result) {
        faces.push_back(face_box);
    }
    output = std::move(faces);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
LibFaceDetector<INPUT, OUTPUT>::LibFaceDetector()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("LIBFACE") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
