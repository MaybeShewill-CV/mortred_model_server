/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: centerface_detector.inl
 * Date: 23-10-18
 ************************************************/

#include "centerface_detector.h"

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
StatusCode CenterFaceDetector<INPUT, OUTPUT>::on_init(const toml::table& params) {
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
        LOG(ERROR) << "unexpected centerface input shape: " << input_info.to_string()
                   << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
const NamedTensor* CenterFaceDetector<INPUT, OUTPUT>::find_output(
    const std::vector<NamedTensor>& outputs, const std::string& name) const {
    for (const auto& item : outputs) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> CenterFaceDetector<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // bgr -> rgb, dynamic resize to a multiple of 32 (the session resizes itself)
    _m_input_size_user = input_image.size();
    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);
    const auto width_resized =
        static_cast<int>(std::ceil(static_cast<float>(input_image.cols) / 32.0f) * 32);
    const auto height_resized =
        static_cast<int>(std::ceil(static_cast<float>(input_image.rows) / 32.0f) * 32);
    cv::resize(tmp, tmp, cv::Size(width_resized, height_resized));
    _m_input_size_host = cv::Size(width_resized, height_resized);
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>({1, 3, height_resized, width_resized});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
StatusCode CenterFaceDetector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                          OUTPUT& output) {
    const auto* heatmap = find_output(outputs, "537");
    const auto* scale = find_output(outputs, "538");
    const auto* offset = find_output(outputs, "539");
    const auto* landmark = find_output(outputs, "540");
    if (heatmap == nullptr || scale == nullptr || offset == nullptr || landmark == nullptr) {
        LOG(ERROR) << "centerface outputs 537/538/539/540 missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    // heatmap layout: [1,1,H,W] over the /4 feature map
    const jinq::models::backend::Tensor& heat_tensor = heatmap->tensor;
    if (heat_tensor.shape.size() != 4) {
        LOG(ERROR) << "unexpected centerface heatmap shape: "
                   << jinq::models::backend::shape_to_string(heat_tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const int output_height = static_cast<int>(heat_tensor.shape[2]);
    const int output_width = static_cast<int>(heat_tensor.shape[3]);
    const int channel_step = output_width * output_height;
    const auto* heat_data = heat_tensor.template data<float>();
    const auto* scale_data = scale->tensor.template data<float>();
    const auto* offset_data = offset->tensor.template data<float>();
    const auto* landmark_data = landmark->tensor.template data<float>();

    std::vector<FaceBBox> decode_result;
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            const int index = h * output_width + w;
            const float score = heat_data[index];
            if (score < _m_score_threshold) {
                continue;
            }
            const float s0 = 4 * std::exp(scale_data[index]);
            const float s1 = 4 * std::exp(scale_data[index + channel_step]);
            const float o0 = offset_data[index];
            const float o1 = offset_data[index + channel_step];

            const float ymin = std::max(0.0f, static_cast<float>(4 * (h + o0 + 0.5) - 0.5 * s0));
            const float xmin = std::max(0.0f, static_cast<float>(4 * (w + o1 + 0.5) - 0.5 * s1));
            const float ymax = std::min(ymin + s0, static_cast<float>(_m_input_size_host.height));
            const float xmax = std::min(xmin + s1, static_cast<float>(_m_input_size_host.width));

            FaceBBox face_info;
            face_info.score = score;
            face_info.bbox.x = xmin;
            face_info.bbox.y = ymin;
            face_info.bbox.width = (xmax - xmin);
            face_info.bbox.height = (ymax - ymin);
            for (int num = 0; num < 5; ++num) {
                cv::Point2f point;
                point.x = s1 * landmark_data[(2 * num + 1) * channel_step + index] + xmin;
                point.y = s0 * landmark_data[(2 * num + 0) * channel_step + index] + ymin;
                face_info.landmarks.push_back(point);
            }
            face_info.class_id = 0;
            decode_result.push_back(face_info);
        }
    }

    // refine bbox coords back into the user image space
    const auto width_scale =
        _m_input_size_user.width / static_cast<float>(_m_input_size_host.width);
    const auto height_scale =
        _m_input_size_user.height / static_cast<float>(_m_input_size_host.height);
    for (auto& face_box : decode_result) {
        face_box.bbox.x *= width_scale;
        face_box.bbox.y *= height_scale;
        face_box.bbox.width *= width_scale;
        face_box.bbox.height *= height_scale;
        for (auto& point : face_box.landmarks) {
            point.x *= width_scale;
            point.y *= height_scale;
        }
    }

    auto nms_result = jinq::common::CvUtils::nms_bboxes(decode_result, _m_nms_threshold);
    if (nms_result.size() > _m_keep_topk) {
        nms_result.resize(_m_keep_topk);
    }
    for (auto& bbox : nms_result) {
        bbox.category = "face";
    }
    FaceOutput faces;
    faces.reserve(nms_result.size());
    for (const auto& bbox : nms_result) {
        faces.push_back(bbox);
    }
    output = std::move(faces);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
CenterFaceDetector<INPUT, OUTPUT>::CenterFaceDetector()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("CENTER_FACE") {}

}
}
}
