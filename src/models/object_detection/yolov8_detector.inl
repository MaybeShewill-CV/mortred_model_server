/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.inl
 * Date: 24-3-13
 ************************************************/

#include "yolov8_detector.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::backend::NamedTensor;
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::on_init(const toml::table& params) {
    if (params.contains("model_score_threshold")) {
        _m_score_threshold = params["model_score_threshold"].value_or<double>(0.0);
    }
    if (params.contains("model_nms_threshold")) {
        _m_nms_threshold = params["model_nms_threshold"].value_or<double>(0.0);
    }
    if (params.contains("model_keep_top_k")) {
        _m_keep_topk = static_cast<int>(params["model_keep_top_k"].value_or<int64_t>(0));
    }
    if (params.contains("model_class_nums")) {
        _m_class_nums = static_cast<int>(params["model_class_nums"].value_or<int64_t>(0));
    }
    if (params.contains("class_names")) {
        const toml::array* cls_names = params["class_names"].as_array();
        if (cls_names == nullptr) {
            LOG(ERROR) << "params key 'class_names' is not an array";
            return StatusCode::MODEL_INIT_FAILED;
        }
        for (size_t idx = 0; idx < cls_names->size(); ++idx) {
            _m_class_id2names[static_cast<int>(idx)] = (*cls_names)[idx].value_or<std::string>("");
        }
    } else {
        for (auto idx = 0; idx < _m_class_nums; ++idx) {
            _m_class_id2names.insert(std::make_pair(idx, ""));
        }
    }

    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected yolov8 input shape: " << input_info.to_string()
                   << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "yolov8 input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> YoloV8Detector<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // bgr -> rgb -> resize -> [0,1] normalize, emitted as f32 nchw
    _m_input_size_user = input_image.size();
    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);
    cv::resize(tmp, tmp, _m_input_size_host);
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 255.0;

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data size mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
cv::Rect2f YoloV8Detector<INPUT, OUTPUT>::transform_bboxes(const cv::Rect2d& bbox) const {
    const auto w_scale = static_cast<float>(_m_input_size_user.width) /
                         static_cast<float>(_m_input_size_host.width);
    const auto h_scale = static_cast<float>(_m_input_size_user.height) /
                         static_cast<float>(_m_input_size_host.height);
    cv::Rect2f result;
    result.x = static_cast<float>(bbox.x * w_scale);
    result.y = static_cast<float>(bbox.y * h_scale);
    result.width = static_cast<float>(bbox.width * w_scale);
    result.height = static_cast<float>(bbox.height * h_scale);
    return result;
}

template<typename INPUT, typename OUTPUT>
StatusCode YoloV8Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                      OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "yolov8 output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    const auto* out_data = tensor.template data<float>();
    if (tensor.shape.size() != 3) {
        LOG(ERROR) << "unexpected yolov8 output shape: "
                   << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto row_size = tensor.shape[1];
    const auto proposal_counts = tensor.shape[2];

    // collect class-filtered candidates in the network (host) coordinate
    // space; NMS runs before mapping the kept boxes back to the input image
    DetectionOutput candidates;
    for (int64_t i = 0; i < proposal_counts; ++i) {
        const float cx = out_data[0 * proposal_counts + i];
        const float cy = out_data[1 * proposal_counts + i];
        const float w = out_data[2 * proposal_counts + i];
        const float h = out_data[3 * proposal_counts + i];

        float cls_score = 0.0f;
        int cls_id = -1;
        // class scores occupy rows 4..row_size-1 (all of them, the last 4
        // classes must not be dropped)
        for (int64_t j = 4; j < row_size; ++j) {
            const float score = out_data[j * proposal_counts + i];
            if (score > cls_score) {
                cls_score = score;
                cls_id = static_cast<int>(j - 4);
            }
        }
        if (cls_score < _m_score_threshold) {
            continue;
        }

        jinq::models::io_define::object_detection::bbox candidate;
        candidate.bbox = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);
        candidate.score = cls_score;
        candidate.class_id = cls_id;
        candidates.push_back(candidate);
    }

    // shared per-class cv::dnn::NMSBoxes suppression used by all detectors
    DetectionOutput nms_result =
        jinq::common::CvUtils::nms_boxes_per_class(candidates, _m_score_threshold, _m_nms_threshold);
    if (nms_result.size() > static_cast<size_t>(_m_keep_topk)) {
        nms_result.resize(static_cast<size_t>(_m_keep_topk));
    }

    // rescale kept boxes from the network space to the original image size
    for (auto& bbox : nms_result) {
        bbox.bbox = transform_bboxes(cv::Rect2d(bbox.bbox));
        const auto name_iter = _m_class_id2names.find(bbox.class_id);
        if (name_iter != _m_class_id2names.end()) {
            bbox.category = name_iter->second;
        }
    }
    output = std::move(nms_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
YoloV8Detector<INPUT, OUTPUT>::YoloV8Detector()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV8") {}

}
}
}
