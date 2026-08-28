/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov6_detector.inl
 * Date: 23-3-3
 ************************************************/

#include "yolov6_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace object_detection {

using DetectionOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::common::CvUtils;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode YoloV6Detector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.4f;
    _m_detection_params.nms_threshold = 0.35f;
    _m_detection_params.keep_top_k = 250;
    _m_detection_params.class_nums = 80;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid yolov6 detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected yolov6 input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "yolov6 input shape has dynamic/invalid H/W: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    cv::Size configured_size;
    if (!parse_model_input_size(params, &configured_size, &param_error) ||
        (params.contains("model_input_image_size") && configured_size != _m_input_size_host)) {
        LOG(ERROR) << "invalid yolov6 input size: " << (param_error.empty() ? "configured size mismatches model input" : param_error);
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> YoloV6Detector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // resize -> bgr2rgb -> [0,1] normalize, emitted as f32 nchw
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
    named.tensor = jinq::models::backend::Tensor::make<float>({1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (input_chw_image_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data size mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), input_chw_image_data.data(), named.tensor.byte_size());
    return {std::move(named)};
}

template <typename INPUT, typename OUTPUT>
StatusCode YoloV6Detector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                      const jinq::models::InferenceContext &context, OUTPUT &output) {
    const auto *output_tensor = jinq::models::backend::find_output(outputs, "outputs");
    if (output_tensor == nullptr) {
        LOG(ERROR) << "yolov6 output tensor 'outputs' is missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = output_tensor->tensor;
    std::string contract_error;
    if (!jinq::models::backend::validate_output_tensor(
            *output_tensor, {jinq::models::backend::DType::F32, 3, {1, -1, _m_detection_params.class_nums + 5}}, &contract_error)) {
        LOG(ERROR) << "yolov6 output contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    const float *output_tensordata = nullptr;
    if (!jinq::models::backend::get_f32_data(tensor, &output_tensordata, &contract_error) ||
        !jinq::models::backend::require_finite_f32(output_tensordata, static_cast<size_t>(tensor.element_count()), output_tensor->name,
                                                   &contract_error)) {
        LOG(ERROR) << "yolov6 output contract failed: " << contract_error;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }
    const auto batch_nums = tensor.shape[0];
    const auto raw_pred_bbox_nums = tensor.shape[1];

    std::vector<std::vector<float>> raw_output;
    raw_output.resize(raw_pred_bbox_nums);
    for (auto &&tmp : raw_output) {
        tmp.resize(_m_detection_params.class_nums + 5, 0.0);
    }
    for (auto index = 0; index < raw_pred_bbox_nums; ++index) {
        for (auto idx = 0; idx < _m_detection_params.class_nums + 5; idx++) {
            raw_output[index][idx] = output_tensordata[index * (_m_detection_params.class_nums + 5) + idx];
        }
    }

    DetectionOutput decode_result;
    for (int batch_num = 0; batch_num < batch_nums; ++batch_num) {
        for (int bbox_index = 0; bbox_index < raw_pred_bbox_nums; ++bbox_index) {
            const std::vector<float> raw_bbox_info = raw_output[bbox_index];
            // thresh bboxes with lower score
            int class_id = -1;
            float max_cls_score = 0.0;
            for (auto cls_idx = 0; cls_idx < _m_detection_params.class_nums; ++cls_idx) {
                if (raw_bbox_info[cls_idx + 5] > max_cls_score) {
                    max_cls_score = raw_bbox_info[cls_idx + 5];
                    class_id = cls_idx;
                }
            }

            const auto bbox_score = raw_bbox_info[4] * max_cls_score;
            if (bbox_score < _m_detection_params.score_threshold) {
                continue;
            }
            // thresh invalid bboxes
            if (raw_bbox_info[2] <= 0 || raw_bbox_info[3] <= 0) {
                continue;
            }

            const auto bbox_area = std::sqrt(raw_bbox_info[2] * raw_bbox_info[3]);
            if (bbox_area < 0 || bbox_area > std::numeric_limits<float>::max()) {
                continue;
            }

            // rescale boxes from img_size to im0 size
            std::vector<float> coords = {raw_bbox_info[0] - raw_bbox_info[2] / 2.0f, raw_bbox_info[1] - raw_bbox_info[3] / 2.0f,
                                         raw_bbox_info[0] + raw_bbox_info[2] / 2.0f, raw_bbox_info[1] + raw_bbox_info[3] / 2.0f};
            const auto w_scale = static_cast<float>(context.source_size.width) / static_cast<float>(_m_input_size_host.width);
            const auto h_scale = static_cast<float>(context.source_size.height) / static_cast<float>(_m_input_size_host.height);
            coords[0] *= w_scale;
            coords[1] *= h_scale;
            coords[2] *= w_scale;
            coords[3] *= h_scale;

            jinq::models::io_define::object_detection::bbox tmp_bbox;
            tmp_bbox.class_id = class_id;
            tmp_bbox.score = bbox_score;
            tmp_bbox.bbox.x = coords[0];
            tmp_bbox.bbox.y = coords[1];
            tmp_bbox.bbox.width = coords[2] - coords[0];
            tmp_bbox.bbox.height = coords[3] - coords[1];
            if (tmp_bbox.bbox.area() < _m_detection_params.min_box_area_px) {
                continue;
            }
            decode_result.push_back(tmp_bbox);
        }
    }

    DetectionOutput nms_result =
        CvUtils::nms_boxes_per_class(decode_result, _m_detection_params.score_threshold, _m_detection_params.nms_threshold);
    if (nms_result.size() > static_cast<size_t>(_m_detection_params.keep_top_k)) {
        nms_result.resize(static_cast<size_t>(_m_detection_params.keep_top_k));
    }
    for (auto &bbox : nms_result) {
        if (bbox.class_id >= 0 && bbox.class_id < static_cast<int>(_m_detection_params.class_names.size())) {
            bbox.category = _m_detection_params.class_names[static_cast<size_t>(bbox.class_id)];
        }
    }
    output = std::move(nms_result);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
YoloV6Detector<INPUT, OUTPUT>::YoloV6Detector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("YOLOV6") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
