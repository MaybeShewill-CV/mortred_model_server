/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: real_esrgan.inl
 * Date: 22-9-29
 ************************************************/

#include "real_esrgan.h"

#include <cstring>

#include <opencv2/opencv.hpp>

#include "glog/logging.h"

namespace jinq {
namespace models {
namespace enhancement {

using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;
using jinq::models::io_define::enhancement::std_enhancement_output;

template <typename INPUT, typename OUTPUT> StatusCode RealEsrGan<INPUT, OUTPUT>::on_init(const toml::table &params) {
    (void)params;
    const auto &inputs = this->session().inputs();
    if (inputs.size() != 1 || inputs.front().shape.size() != 4 || inputs.front().shape[3] != 3) {
        LOG(ERROR) << "unexpected real esrgan input io: " << (inputs.empty() ? std::string("empty") : inputs.front().to_string())
                   << ", expected [N,H,W,3]";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(inputs.front().shape[1]);
    _m_input_size_host.width = static_cast<int>(inputs.front().shape[2]);
    // dynamic input (unset mnn dims): the size is resolved per run in preprocess
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> RealEsrGan<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    if (input_image.size().height < 10 || input_image.size().width < 10 || (input_image.channels() != 3 && input_image.channels() != 4)) {
        LOG(ERROR) << "invalid real esrgan image size or channels: " << input_image.size() << ", channels=" << input_image.channels();
        return {};
    }

    cv::Mat output_src;
    input_image.copyTo(output_src);
    if (output_src.channels() == 4) {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(output_src, output_src, cv::COLOR_BGR2RGB);
    }
    if (output_src.type() != CV_32FC3) {
        output_src.convertTo(output_src, CV_32FC3);
    }
    output_src /= 255.0;

    std::vector<NamedTensor> tensors;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>({1, output_src.rows, output_src.cols, 3});
    const auto bytes = output_src.total() * output_src.elemSize();
    if (bytes != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed real esrgan image byte size mismatches tensor";
        return {};
    }
    if (output_src.isContinuous()) {
        std::memcpy(named.tensor.buffer.data(), output_src.data, bytes);
    } else {
        uint8_t *dst = named.tensor.buffer.data();
        for (int row = 0; row < output_src.rows; ++row) {
            const auto row_bytes = static_cast<size_t>(output_src.cols) * output_src.elemSize();
            std::memcpy(dst, output_src.ptr(row), row_bytes);
            dst += row_bytes;
        }
    }
    tensors.push_back(std::move(named));
    return tensors;
}

template <typename INPUT, typename OUTPUT>
StatusCode RealEsrGan<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                  const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) {
    if (outputs.empty()) {
        LOG(ERROR) << "real esrgan output tensor is empty";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto &tensor = outputs.front().tensor;
    // the exported mnn output is nchw ([1,3,H,W]); the old TENSORFLOW host
    // wrapper reordered it to hwc, do the same here
    if (tensor.shape.size() != 4 || tensor.shape[0] != 1 || tensor.shape[1] != 3) {
        LOG(ERROR) << "unexpected real esrgan output shape: " << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto output_h = static_cast<int>(tensor.shape[2]);
    const auto output_w = static_cast<int>(tensor.shape[3]);
    if (output_w <= 0 || output_h <= 0 || tensor.element_count() != 3 * output_w * output_h) {
        LOG(ERROR) << "invalid real esrgan output shape: " << jinq::models::backend::shape_to_string(tensor.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto *host_data = tensor.template data<float>();
    std::vector<uchar> output_img_data(static_cast<size_t>(tensor.element_count()));
    for (int64_t index = 0; index < tensor.element_count(); ++index) {
        auto pix_val_f = host_data[index] * 255.0;
        if (pix_val_f < 0.0) {
            pix_val_f = 0.0;
        }
        if (pix_val_f >= 255) {
            pix_val_f = 255.0;
        }
        output_img_data[static_cast<size_t>(index)] = static_cast<uchar>(pix_val_f);
    }

    auto hwc_data = jinq::common::CvUtils::convert_to_hwc_vec<uchar>(output_img_data, 3, output_h, output_w);
    std_enhancement_output internal_out;
    cv::Mat result_image(cv::Size(output_w, output_h), CV_8UC3, hwc_data.data());
    cv::cvtColor(result_image, internal_out.enhancement_result, cv::COLOR_RGB2BGR);
    output = std::move(internal_out);
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
RealEsrGan<INPUT, OUTPUT>::RealEsrGan() : jinq::models::BackendCvModel<INPUT, OUTPUT>("REALESRGAN") {}

} // namespace enhancement
} // namespace models
} // namespace jinq
