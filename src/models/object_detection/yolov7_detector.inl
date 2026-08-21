/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov7_detector.inl
 * Date: 22-7-14
 ************************************************/

#include "yolov7_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::backend::NamedTensor;
using jinq::common::CvUtils;
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
StatusCode YoloV7Detector<INPUT, OUTPUT>::on_init(const toml::table& params) {
    if (params.contains("model_score_threshold")) {
        _m_score_threshold = params["model_score_threshold"].value_or<double>(0.0);
    }
    if (params.contains("model_nms_threshold")) {
        _m_nms_threshold = params["model_nms_threshold"].value_or<double>(0.0);
    }
    if (params.contains("model_keep_top_k")) {
        _m_keep_topk = static_cast<long>(params["model_keep_top_k"].value_or<int64_t>(0));
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
            _m_class_id2names.insert(
                std::make_pair(static_cast<int>(idx), (*cls_names)[idx].value_or<std::string>("")));
        }
    } else {
        for (auto idx = 0; idx < _m_class_nums; ++idx) {
            _m_class_id2names.insert(std::make_pair(idx, ""));
        }
    }

    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected yolov7 input shape: " << input_info.to_string()
                   << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "yolov7 input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> YoloV7Detector<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // resize -> bgr2rgb -> [0,1] normalize, emitted as f32 nchw
    _m_input_size_user = input_image.size();
    cv::Mat tmp;
    cv::resize(input_image, tmp, _m_input_size_host);
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 255.0;

    const auto input_chw_image_data = CvUtils::convert_to_chw_vec(tmp);
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (input_chw_image_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data size mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), input_chw_image_data.data(), named.tensor.byte_size());
    return {std::move(named)};
}

template<typename INPUT, typename OUTPUT>
const NamedTensor* YoloV7Detector<INPUT, OUTPUT>::find_output(
    const std::vector<NamedTensor>& outputs, const std::string& name) const {
    const auto iter = std::find_if(
        outputs.begin(), outputs.end(),
        [&name](const NamedTensor& item) { return item.name == name; });
    return iter == outputs.end() ? nullptr : &*iter;
}

template<typename INPUT, typename OUTPUT>
StatusCode YoloV7Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                       OUTPUT& output) {
    // yolov7.mnn exports three raw output heads [1, 3, H, W, 85]:
    //   "output" -> 80x80 (stride 8), "518" -> 40x40 (stride 16),
    //   "532" -> 20x20 (stride 32)
    const std::array<const NamedTensor*, 3> heads = {
        find_output(outputs, "output"), find_output(outputs, "518"),
        find_output(outputs, "532")};
    if (std::any_of(heads.begin(), heads.end(),
                    [](const NamedTensor* head) { return head == nullptr; })) {
        LOG(ERROR) << "yolov7 output heads 'output', '518' or '532' are missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    const int strides[3] = {8, 16, 32};
    const float anchors[3][3][2] = {
        {{12, 16}, {19, 36}, {40, 28}},
        {{36, 75}, {76, 55}, {72, 146}},
        {{142, 110}, {192, 243}, {459, 401}},
    };

    DetectionOutput decode_result;
    for (std::size_t hi = 0; hi < heads.size(); ++hi) {
        const auto& shape = heads[hi]->tensor.shape;
        if (shape.size() != 5) {
            continue;
        }
        const int anchor_nums = static_cast<int>(shape[1]);
        const int grid_h = static_cast<int>(shape[2]);
        const int grid_w = static_cast<int>(shape[3]);
        const int attrs = static_cast<int>(shape[4]);
        const float* data = heads[hi]->tensor.template data<float>();
        const int stride = strides[hi];

        for (int a = 0; a < anchor_nums && a < 3; ++a) {
            const float anchor_w = anchors[hi][a][0];
            const float anchor_h = anchors[hi][a][1];
            for (int row = 0; row < grid_h; ++row) {
                for (int col = 0; col < grid_w; ++col) {
                    const float* p = data + (((a * grid_h + row) * grid_w + col) * attrs);
                    const float obj_score = sigmoid(p[4]);
                    if (obj_score < 0.05f) {
                        continue;
                    }
                    int class_id = -1;
                    float max_cls_score = 0.0f;
                    for (int c = 5; c < attrs; ++c) {
                        const float cls_score = sigmoid(p[c]);
                        if (cls_score > max_cls_score) {
                            max_cls_score = cls_score;
                            class_id = c - 5;
                        }
                    }
                    const float bbox_score = obj_score * max_cls_score;
                    if (bbox_score < _m_score_threshold) {
                        continue;
                    }
                    const float center_x = (2.0f * sigmoid(p[0]) - 0.5f + col) * stride;
                    const float center_y = (2.0f * sigmoid(p[1]) - 0.5f + row) * stride;
                    const float box_w = std::pow(2.0f * sigmoid(p[2]), 2.0f) * anchor_w;
                    const float box_h = std::pow(2.0f * sigmoid(p[3]), 2.0f) * anchor_h;
                    if (box_w <= 0.0f || box_h <= 0.0f) {
                        continue;
                    }
                    jinq::models::io_define::object_detection::bbox tmp_bbox;
                    tmp_bbox.class_id = class_id;
                    tmp_bbox.score = bbox_score;
                    tmp_bbox.bbox.x = center_x - box_w / 2.0f;
                    tmp_bbox.bbox.y = center_y - box_h / 2.0f;
                    tmp_bbox.bbox.width = box_w;
                    tmp_bbox.bbox.height = box_h;
                    if (tmp_bbox.bbox.area() < 5) {
                        continue;
                    }
                    decode_result.push_back(tmp_bbox);
                }
            }
        }
    }

    // rescale boxes from 640-space to the original image size
    const auto w_scale = static_cast<float>(_m_input_size_user.width) /
                         static_cast<float>(_m_input_size_host.width);
    const auto h_scale = static_cast<float>(_m_input_size_user.height) /
                         static_cast<float>(_m_input_size_host.height);
    for (auto& bbox : decode_result) {
        bbox.bbox.x *= w_scale;
        bbox.bbox.y *= h_scale;
        bbox.bbox.width *= w_scale;
        bbox.bbox.height *= h_scale;
    }

    DetectionOutput nms_result = CvUtils::nms_bboxes(decode_result, _m_nms_threshold);
    if (nms_result.size() > static_cast<size_t>(_m_keep_topk)) {
        nms_result.resize(static_cast<size_t>(_m_keep_topk));
    }
    for (auto& bbox : nms_result) {
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
YoloV7Detector<INPUT, OUTPUT>::YoloV7Detector()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV7") {}

}
}
}
