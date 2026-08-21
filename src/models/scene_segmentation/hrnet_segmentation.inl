/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: hrnet_segmentation.inl
 * Date: 23-11-17
 ************************************************/

#include "hrnet_segmentation.h"

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
StatusCode HRNetSegmentation<INPUT, OUTPUT>::on_init(const toml::table& params) {
    (void)params;
    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3 ||
        input_info.dynamic) {
        LOG(ERROR) << "unexpected hrnet input shape: " << input_info.to_string()
                   << ", expected static [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    const auto& output_info = this->session().outputs().front();
    if (output_info.dtype != jinq::models::backend::DType::I32) {
        LOG(ERROR) << "unexpected hrnet argmax output dtype: " << output_info.to_string();
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template<typename INPUT, typename OUTPUT>
std::vector<NamedTensor> HRNetSegmentation<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // bgr -> rgb -> resize -> [0,1] -> (x-0.5)/0.5, f32 nchw
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

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(tmp);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, _m_input_size_host.height, _m_input_size_host.width});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template<typename INPUT, typename OUTPUT>
StatusCode HRNetSegmentation<INPUT, OUTPUT>::postprocess(
    const std::vector<NamedTensor>& outputs, OUTPUT& output) {
    if (outputs.empty()) {
        LOG(ERROR) << "hrnet output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& tensor = outputs.front().tensor;
    const auto* mask_data = tensor.template data<int32_t>();
    if (tensor.element_count() < static_cast<int64_t>(_m_input_size_host.area())) {
        LOG(ERROR) << "hrnet mask smaller than the input map: "
                   << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    cv::Mat result_image(_m_input_size_host, CV_32SC1,
                         const_cast<int32_t*>(reinterpret_cast<const int32_t*>(mask_data)));
    cv::Mat resized;
    cv::resize(result_image, resized, _m_input_size_user, 0.0, 0.0, cv::INTER_NEAREST);
    SegmentationOutput internal_out;
    internal_out.segmentation_result = std::move(resized);
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template<typename INPUT, typename OUTPUT>
HRNetSegmentation<INPUT, OUTPUT>::HRNetSegmentation()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("HRNET_SEGMENTATION") {}

}
}
}
