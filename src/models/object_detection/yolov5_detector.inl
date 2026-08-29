/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov5_detector.inl
 * Date: 22-6-7
 ************************************************/

#include "yolov5_detector.h"

#include <utility>

#include "glog/logging.h"
#include "models/object_detection/detector_common.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode YoloV5Detector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.4f;
    _m_detection_params.nms_threshold = 0.35f;
    _m_detection_params.keep_top_k = 250;
    _m_detection_params.class_nums = 80;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid yolov5 detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected yolov5 input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "yolov5 input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    cv::Size configured_size;
    if (!parse_model_input_size(params, &configured_size, &param_error)) {
        LOG(ERROR) << "invalid yolov5 input size: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (params.contains("model_input_image_size") && configured_size != _m_input_size_host) {
        LOG(ERROR) << "yolov5 model_input_image_size is " << configured_size << ", but model input is " << _m_input_size_host;
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> YoloV5Detector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // resize -> bgr2rgb -> [0,1] normalize, emitted as f32 nchw
    cv::Mat tmp;
    cv::resize(input_image, tmp, _m_input_size_host);
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 255.0;

    NamedTensor named;
    if (!make_nchw_input(this->session().inputs().front().name, tmp, &named)) {
        return {};
    }
    return {std::move(named)};
}

template <typename INPUT, typename OUTPUT>
StatusCode YoloV5Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                      const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    F32OutputView output_view;
    const auto output_status = validated_f32_output(
        outputs, "output", {jinq::models::backend::DType::F32, 3, {1, -1, _m_detection_params.class_nums + 5}}, "yolov5", &output_view);
    if (output_status != StatusCode::OK) {
        return output_status;
    }
    const auto &tensor = *output_view.tensor;
    const float *output_tensordata = output_view.data;
    const auto batch_nums = tensor.shape[0];
    const auto raw_pred_bbox_nums = tensor.shape[1];
    const size_t row_size = static_cast<size_t>(_m_detection_params.class_nums + 5);

    DetectionGeometryScale geometry_scale;
    std::string geometry_error;
    if (!make_detection_geometry_scale(context, &geometry_scale, &geometry_error)) {
        LOG(ERROR) << "yolov5 " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    DetectionOutput decode_result;
    for (int batch_num = 0; batch_num < batch_nums; ++batch_num) {
        const size_t batch_offset = batch_num * raw_pred_bbox_nums * row_size;
        for (int bbox_index = 0; bbox_index < raw_pred_bbox_nums; ++bbox_index) {
            const size_t offset = batch_offset + bbox_index * row_size;
            // thresh bboxes with lower score
            int class_id = -1;
            float max_cls_score = 0.0;
            for (auto cls_idx = 0; cls_idx < _m_detection_params.class_nums; ++cls_idx) {
                const float cls_score = output_tensordata[offset + cls_idx + 5];
                if (cls_score > max_cls_score) {
                    max_cls_score = cls_score;
                    class_id = cls_idx;
                }
            }

            const float obj_score = output_tensordata[offset + 4];
            const auto bbox_score = obj_score * max_cls_score;
            if (bbox_score < _m_detection_params.score_threshold) {
                continue;
            }

            const float box_w = output_tensordata[offset + 2];
            const float box_h = output_tensordata[offset + 3];
            // thresh invalid bboxes
            if (box_w <= 0 || box_h <= 0) {
                continue;
            }

            jinq::models::io_define::object_detection::bbox tmp_bbox;
            tmp_bbox.class_id = class_id;
            tmp_bbox.score = bbox_score;
            tmp_bbox.bbox = scale_detection_bbox(
                {output_tensordata[offset + 0] - box_w / 2.0f, output_tensordata[offset + 1] - box_h / 2.0f, box_w, box_h}, geometry_scale);
            if (tmp_bbox.bbox.area() < _m_detection_params.min_box_area_px) {
                continue;
            }
            decode_result.push_back(tmp_bbox);
        }
    }

    DetectionOutput nms_result = finalize_detections(std::move(decode_result), _m_detection_params);
    output = std::move(nms_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
YoloV5Detector<INPUT, OUTPUT>::YoloV5Detector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV5") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
