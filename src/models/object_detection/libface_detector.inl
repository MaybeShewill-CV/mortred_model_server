/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.inl
 * Date: 22-6-10
 ************************************************/

#include "libface_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "models/backend/model_runtime.h"
#include "models/object_detection/detector_common.h"

namespace jinq {
namespace models {
namespace object_detection {

using FaceBBox = jinq::models::io_define::object_detection::face_bbox;
using FaceOutput = jinq::models::io_define::object_detection::std_face_detection_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

namespace {

constexpr int kYuNetAlign = 32;
constexpr std::array<int, 3> kYuNetStrides = {8, 16, 32};

inline cv::Size align_to_divisor(const cv::Size &size, int divisor) {
    if (divisor <= 0 || size.width <= 0 || size.height <= 0) {
        return {};
    }
    return {((size.width + divisor - 1) / divisor) * divisor, ((size.height + divisor - 1) / divisor) * divisor};
}

inline float clamp01(float value) { return std::min(1.0f, std::max(0.0f, value)); }

} // namespace

template <typename INPUT, typename OUTPUT> StatusCode LibFaceDetector<INPUT, OUTPUT>::on_init(const toml::table &params) {
    _m_detection_params.score_threshold = 0.6f;
    _m_detection_params.nms_threshold = 0.3f;
    _m_detection_params.keep_top_k = 250;
    std::string param_error;
    if (!_m_detection_params.parse(params, &_m_detection_params, &param_error)) {
        LOG(ERROR) << "invalid libface detection params: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto &input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3) {
        LOG(ERROR) << "unexpected libface input shape: " << input_info.to_string() << ", expected [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (!parse_model_input_size(params, &_m_input_size_host, &param_error)) {
        LOG(ERROR) << "invalid libface input size: " << param_error;
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (_m_input_size_host.area() > 0) {
        this->set_image_decode_hint(_m_input_size_host, parse_image_decode_upscale(params, "libface"));
    } else {
        this->set_image_decode_hint(cv::Size(), 1.0f);
    }
    // Stretch to toml [H,W] (or native size), then right/bottom zero-pad /32.
    // Path B still needs type=tensorrt at serving time; onnx stays CPU decode.
    this->set_gpu_preprocess({
        .resize = jinq::models::backend::GpuPreprocessDescriptor::Resize::DIRECT_RESIZE_PAD_TO_MULTIPLE,
        .norm = {.scale = 1.0f},
        .color = jinq::models::backend::GpuPreprocessDescriptor::Color::BGR,
        .pad_value = 0,
        .output_dtype = input_info.dtype,
        .output_nhwc = false,
        .pre_crop_size = _m_input_size_host,
        .align_multiple = kYuNetAlign,
        .dynamic_size = true,
    });

    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> LibFaceDetector<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    cv::Mat working;
    if (_m_input_size_host.area() > 0) {
        cv::resize(input_image, working, _m_input_size_host, 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        working = input_image;
    }
    const cv::Size padded = align_to_divisor(working.size(), kYuNetAlign);
    if (padded.area() <= 0) {
        LOG(ERROR) << "libface failed to align input size " << working.size();
        return {};
    }
    if (padded != working.size()) {
        cv::Mat canvas;
        cv::copyMakeBorder(working, canvas, 0, padded.height - working.rows, 0, padded.width - working.cols, cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));
        working = std::move(canvas);
    }
    auto result = jinq::models::backend::ImagePipeline(working).to_float().nchw(this->session().inputs().front());
    if (!result.ok()) {
        LOG(ERROR) << result.error;
        return {};
    }
    return {std::move(result.value)};
}

template <typename INPUT, typename OUTPUT>
StatusCode LibFaceDetector<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                       const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    const int pad_w = context.network_size.width;
    const int pad_h = context.network_size.height;
    if (pad_w <= 0 || pad_h <= 0 || pad_w % kYuNetAlign != 0 || pad_h % kYuNetAlign != 0) {
        LOG(ERROR) << "libface network size must be a positive multiple of " << kYuNetAlign << ", got " << context.network_size;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }

    const cv::Size unpadded = _m_input_size_host.area() > 0 ? _m_input_size_host : context.source_size;
    if (unpadded.width <= 0 || unpadded.height <= 0) {
        LOG(ERROR) << "libface invalid unpadded size";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    if (pad_w < unpadded.width || pad_h < unpadded.height) {
        LOG(ERROR) << "libface padded network " << context.network_size << " is smaller than unpadded " << unpadded;
        return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
    }

    std::vector<FaceBBox> decode_result;
    for (const int stride : kYuNetStrides) {
        const std::string stride_s = std::to_string(stride);
        const auto *cls = jinq::models::backend::find_output(outputs, "cls_" + stride_s);
        const auto *obj = jinq::models::backend::find_output(outputs, "obj_" + stride_s);
        const auto *bbox = jinq::models::backend::find_output(outputs, "bbox_" + stride_s);
        const auto *kps = jinq::models::backend::find_output(outputs, "kps_" + stride_s);
        if (cls == nullptr || obj == nullptr || bbox == nullptr || kps == nullptr) {
            LOG(ERROR) << "libface YuNet heads missing for stride " << stride;
            return StatusCode::MODEL_EMPTY_OUTPUT;
        }

        const int cols = pad_w / stride;
        const int rows = pad_h / stride;
        const int64_t anchors = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
        std::string contract_error;
        if (!jinq::models::backend::validate_output_tensor(*cls, {jinq::models::backend::DType::F32, 3, {1, anchors, 1}},
                                                           &contract_error) ||
            !jinq::models::backend::validate_output_tensor(*obj, {jinq::models::backend::DType::F32, 3, {1, anchors, 1}},
                                                           &contract_error) ||
            !jinq::models::backend::validate_output_tensor(*bbox, {jinq::models::backend::DType::F32, 3, {1, anchors, 4}},
                                                           &contract_error) ||
            !jinq::models::backend::validate_output_tensor(*kps, {jinq::models::backend::DType::F32, 3, {1, anchors, 10}},
                                                           &contract_error)) {
            LOG(ERROR) << "libface output contract failed at stride " << stride << ": " << contract_error;
            return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
        }

        const float *cls_data = nullptr;
        const float *obj_data = nullptr;
        const float *bbox_data = nullptr;
        const float *kps_data = nullptr;
        if (!jinq::models::backend::get_f32_data(cls->tensor, &cls_data, &contract_error) ||
            !jinq::models::backend::get_f32_data(obj->tensor, &obj_data, &contract_error) ||
            !jinq::models::backend::get_f32_data(bbox->tensor, &bbox_data, &contract_error) ||
            !jinq::models::backend::get_f32_data(kps->tensor, &kps_data, &contract_error) ||
            !jinq::models::backend::require_finite_f32(cls_data, cls->tensor.element_count(), cls->name, &contract_error) ||
            !jinq::models::backend::require_finite_f32(obj_data, obj->tensor.element_count(), obj->name, &contract_error) ||
            !jinq::models::backend::require_finite_f32(bbox_data, bbox->tensor.element_count(), bbox->name, &contract_error) ||
            !jinq::models::backend::require_finite_f32(kps_data, kps->tensor.element_count(), kps->name, &contract_error)) {
            LOG(ERROR) << "libface output contract failed at stride " << stride << ": " << contract_error;
            return StatusCode::MODEL_OUTPUT_CONTRACT_FAILED;
        }

        const float stride_f = static_cast<float>(stride);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const size_t idx = static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
                const float score = std::sqrt(clamp01(cls_data[idx]) * clamp01(obj_data[idx]));
                if (score < _m_detection_params.score_threshold) {
                    continue;
                }

                const float cx = (static_cast<float>(col) + bbox_data[idx * 4 + 0]) * stride_f;
                const float cy = (static_cast<float>(row) + bbox_data[idx * 4 + 1]) * stride_f;
                const float width = std::exp(bbox_data[idx * 4 + 2]) * stride_f;
                const float height = std::exp(bbox_data[idx * 4 + 3]) * stride_f;

                FaceBBox face_box;
                face_box.score = score;
                face_box.bbox = cv::Rect2f(cx - width * 0.5f, cy - height * 0.5f, width, height);
                face_box.landmarks.reserve(5);
                for (int landmark = 0; landmark < 5; ++landmark) {
                    const float px = (kps_data[idx * 10 + 2 * landmark] + static_cast<float>(col)) * stride_f;
                    const float py = (kps_data[idx * 10 + 2 * landmark + 1] + static_cast<float>(row)) * stride_f;
                    face_box.landmarks.emplace_back(px, py);
                }
                face_box.class_id = 0;
                decode_result.push_back(std::move(face_box));
            }
        }
    }

    auto nms_result = finalize_detections(std::move(decode_result), _m_detection_params, context);
    const float scale_x = static_cast<float>(context.source_size.width) / static_cast<float>(unpadded.width);
    const float scale_y = static_cast<float>(context.source_size.height) / static_cast<float>(unpadded.height);
    for (auto &face_box : nms_result) {
        face_box.bbox.x *= scale_x;
        face_box.bbox.y *= scale_y;
        face_box.bbox.width *= scale_x;
        face_box.bbox.height *= scale_y;
        for (auto &landmark : face_box.landmarks) {
            landmark.x *= scale_x;
            landmark.y *= scale_y;
        }
        face_box.category = "face";
    }

    FaceOutput faces;
    faces.reserve(nms_result.size());
    for (const auto &face_box : nms_result) {
        faces.push_back(face_box);
    }
    output = std::move(faces);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
LibFaceDetector<INPUT, OUTPUT>::LibFaceDetector() : jinq::models::BackendCvModel<INPUT, OUTPUT>("LIBFACE") {}

} // namespace object_detection
} // namespace models
} // namespace jinq
