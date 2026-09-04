/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov7_detector.inl
 * Date: 22-7-14
 ************************************************/

#include "yolov7_detector.h"

#include <algorithm>
#include <cmath>

#include "glog/logging.h"
#include "models/backend/model_runtime.h"
#include "models/object_detection/detector_common.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode YoloV7Detector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.4f;
    _m_detection_params.nms_threshold = 0.35f;
    _m_detection_params.keep_top_k = 250;
    _m_detection_params.class_nums = 80;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid yolov7 detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected yolov7 input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "yolov7 input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    cv::Size configured_size;
    if (!parse_model_input_size(params, &configured_size, &param_error) ||
        (params.contains("model_input_image_size") && configured_size != _m_input_size_host)) {
        LOG(ERROR) << "invalid yolov7 input size: " << (param_error.empty() ? "configured size mismatches model input" : param_error);
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> YoloV7Detector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // resize / colour / normalize, emitted as f32 nchw
    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .resize(_m_input_size_host)
                      .bgr_to_rgb()
                      .to_float()
                      .scale(1.0f / 255.0f)
                      .nchw(this->session().inputs().front().name);
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode YoloV7Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                      const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    // yolov7.mnn exports three raw output heads [1, 3, H, W, 85]:
    //   "output" -> 80x80 (stride 8), "518" -> 40x40 (stride 16),
    //   "532" -> 20x20 (stride 32)
    const std::array<const NamedTensor *, 3> heads = {jinq::models::backend::find_output(outputs, "output"),
                                                      jinq::models::backend::find_output(outputs, "518"),
                                                      jinq::models::backend::find_output(outputs, "532")};
    if (std::any_of(heads.begin(), heads.end(), [](const NamedTensor *head) { return head == nullptr; })) {
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

    GeometryScale geometry_scale;
    std::string geometry_error;
    if (!backend::make_geometry_scale(context, &geometry_scale, &geometry_error)) {
        LOG(ERROR) << "yolov7 " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    DetectionOutput decode_result;
    const auto contract_attrs = _m_detection_params.class_nums + 5;
    for (std::size_t hi = 0; hi < heads.size(); ++hi) {
        const auto &shape = heads[hi]->tensor.shape;
        std::string contract_error;
        if (!jinq::models::backend::validate_output_tensor(
                *heads[hi], {jinq::models::backend::DType::F32, 5, {1, 3, -1, -1, contract_attrs}}, &contract_error)) {
            LOG(ERROR) << "yolov7 output contract failed for head " << hi << ": " << contract_error;
            return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
        }
        const int anchor_nums = static_cast<int>(shape[1]);
        const int grid_h = static_cast<int>(shape[2]);
        const int grid_w = static_cast<int>(shape[3]);
        const int attrs = static_cast<int>(shape[4]);
        const float *data = nullptr;
        if (!jinq::models::backend::get_f32_data(heads[hi]->tensor, &data, &contract_error) ||
            !jinq::models::backend::require_finite_f32(data, heads[hi]->tensor.element_count(), heads[hi]->name, &contract_error)) {
            LOG(ERROR) << "yolov7 output contract failed for head " << hi << ": " << contract_error;
            return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
        }
        const int stride = strides[hi];

        for (int a = 0; a < anchor_nums && a < 3; ++a) {
            const float anchor_w = anchors[hi][a][0];
            const float anchor_h = anchors[hi][a][1];
            for (int row = 0; row < grid_h; ++row) {
                for (int col = 0; col < grid_w; ++col) {
                    const float *p = data + (((a * grid_h + row) * grid_w + col) * attrs);
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
                    if (bbox_score < _m_detection_params.score_threshold) {
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
                    if (tmp_bbox.bbox.area() < _m_detection_params.min_box_area_px) {
                        continue;
                    }
                    decode_result.push_back(tmp_bbox);
                }
            }
        }
    }

    for (auto &bbox : decode_result) {
        bbox.bbox = backend::scale_bbox(bbox.bbox, geometry_scale);
    }

    DetectionOutput nms_result = finalize_detections(std::move(decode_result), _m_detection_params, context);
    output = std::move(nms_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
YoloV7Detector<INPUT, OUTPUT>::YoloV7Detector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV7") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
