/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: msocrnet.inl
 * Date: 23-3-11
 ************************************************/

#include "msocrnet.h"

#include <cstring>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

using SegmentationOutput =
    jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
using jinq::models::backend::NamedTensor;
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
StatusCode MsOcrNet<INPUT, OUTPUT>::on_init(const toml::table& params) {
    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4) {
        LOG(ERROR) << "unexpected msocrnet input shape: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (input_info.shape[3] == 3) {
        _m_input_is_nhwc = true;
        _m_input_size_host.height = static_cast<int>(input_info.shape[1]);
        _m_input_size_host.width = static_cast<int>(input_info.shape[2]);
    } else if (input_info.shape[1] == 3) {
        _m_input_is_nhwc = false;
        _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
        _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    } else {
        LOG(ERROR) << "cannot locate channel dim in msocrnet input: " << input_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_input_size_host.area() <= 0 && params.contains("model_input_image_size")) {
        const toml::array* size = params["model_input_image_size"].as_array();
        if (size != nullptr && size->size() == 2) {
            _m_input_size_host.height = static_cast<int>((*size)[0].value_or<int64_t>(0));
            _m_input_size_host.width = static_cast<int>((*size)[1].value_or<int64_t>(0));
        }
    }
    if (_m_input_size_host.area() <= 0) {
        LOG(ERROR) << "msocrnet input size unresolved (dynamic input requires "
                      "params.model_input_image_size)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> MsOcrNet<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // bgr -> rgb -> resize -> [0,1] -> (x-0.5)/0.5
    _m_input_size_user = input_image.size();
    cv::Mat tmp;
    cv::cvtColor(input_image, tmp, cv::COLOR_BGR2RGB);
    if (tmp.size() != _m_input_size_host) {
        cv::resize(tmp, tmp, _m_input_size_host);
    }
    if (tmp.type() != CV_32FC3) {
        tmp.convertTo(tmp, CV_32FC3);
    }
    tmp /= 255.0;
    cv::subtract(tmp, cv::Scalar(0.5, 0.5, 0.5), tmp);
    cv::divide(tmp, cv::Scalar(0.5, 0.5, 0.5), tmp);

    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    if (_m_input_is_nhwc) {
        named.tensor = jinq::models::backend::Tensor::make<float>(
            {1, _m_input_size_host.height, _m_input_size_host.width, 3});
        const auto bytes = tmp.total() * tmp.elemSize();
        if (bytes != named.tensor.byte_size()) {
            LOG(ERROR) << "preprocessed image bytes mismatch tensor size";
            return {};
        }
        std::memcpy(named.tensor.buffer.data(), tmp.data, bytes);
    } else {
        const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
        named.tensor = jinq::models::backend::Tensor::make<float>(
            {1, 3, _m_input_size_host.height, _m_input_size_host.width});
        if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
            LOG(ERROR) << "preprocessed chw data mismatches tensor size";
            return {};
        }
        std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    }
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
StatusCode MsOcrNet<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "msocrnet output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    if (tensor.dtype != jinq::models::backend::DType::I32 &&
        tensor.dtype != jinq::models::backend::DType::I64) {
        LOG(ERROR) << "msocrnet argmax output dtype must be i32/i64, got " << tensor.dtype;
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    // argmax mask layout: [1,H,W] (onnx) or [1,H,W,1] (mnn)
    if (tensor.shape.size() < 3) {
        LOG(ERROR) << "unexpected msocrnet output shape: "
                   << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto height = tensor.shape[1];
    const auto width = tensor.shape[2];
    const auto element_count = height * width;
    if (element_count <= 0) {
        LOG(ERROR) << "msocrnet output mask is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    cv::Mat seg_mask(_m_input_size_host, CV_32SC1, cv::Scalar(0));
    if (tensor.dtype == jinq::models::backend::DType::I32) {
        const auto* data = tensor.template data<int32_t>();
        for (int64_t idx = 0; idx < element_count; ++idx) {
            seg_mask.at<int32_t>(static_cast<int>(idx)) = data[idx];
        }
    } else {
        const auto* data = tensor.template data<int64_t>();
        for (int64_t idx = 0; idx < element_count; ++idx) {
            seg_mask.at<int32_t>(static_cast<int>(idx)) = static_cast<int32_t>(data[idx]);
        }
    }
    cv::Mat resized_mask;
    cv::resize(seg_mask, resized_mask, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);

    SegmentationOutput internal_out;
    internal_out.segmentation_result = resized_mask.clone();
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
MsOcrNet<INPUT, OUTPUT>::MsOcrNet()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("MSOCRNET") {}

}
}
}
