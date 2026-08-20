/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: fast_sam_segmentor.cpp
 * Date: 23-9-14
 ************************************************/

#include "fast_sam_segmentor.h"

#include <algorithm>
#include <cstring>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/cv_utils.h"

namespace jinq {
namespace models {
namespace segment_anything {

using jinq::models::backend::NamedTensor;
using jinq::common::CvUtils;
using jinq::common::StatusCode;

namespace {

struct PredsBBox {
    cv::Rect2f bbox;
    float score = 0.0f;
    int class_id = 0;
    std::vector<float> masks;
};

}  // namespace

template <typename INPUT, typename OUTPUT>
StatusCode FastSamSegmentor<INPUT, OUTPUT>::on_init(const toml::table& params) {
    if (params.contains("conf_thresh")) {
        _m_conf_thresh = params["conf_thresh"].value_or<double>(0.0);
    }
    if (params.contains("iou_thresh")) {
        _m_iou_thresh = params["iou_thresh"].value_or<double>(0.0);
    }

    const auto& input_info = this->session().inputs().front();
    if (input_info.shape.size() != 4 || input_info.shape[1] != 3 || input_info.dynamic) {
        LOG(ERROR) << "unexpected fastsam input shape: " << input_info.to_string()
                   << ", expected static [N,3,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_tensor_size.height = static_cast<int>(input_info.shape[2]);
    _m_input_tensor_size.width = static_cast<int>(input_info.shape[3]);

    bool has_output1 = false;
    for (const auto& info : this->session().outputs()) {
        if (info.name == "output1") {
            has_output1 = true;
            if (info.shape.size() == 4) {
                _m_preds_mask_size.height = static_cast<int>(info.shape[2]);
                _m_preds_mask_size.width = static_cast<int>(info.shape[3]);
            }
        }
    }
    if (!has_output1 || _m_preds_mask_size.area() <= 0) {
        LOG(ERROR) << "fastsam mask proto output 'output1' missing or invalid";
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
const NamedTensor* FastSamSegmentor<INPUT, OUTPUT>::find_output(
    const std::vector<NamedTensor>& outputs, const std::string& name) const {
    for (const auto& item : outputs) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

template <typename INPUT, typename OUTPUT>
std::vector<NamedTensor> FastSamSegmentor<INPUT, OUTPUT>::preprocess(const cv::Mat& input_image) {
    // rgb -> long-side scale -> right/bottom zero pad -> [0,1] (f32 nchw)
    _m_input_image_size = input_image.size();
    const auto input_node_h = _m_input_tensor_size.height;
    const auto input_node_w = _m_input_tensor_size.width;
    const auto long_side = std::max(_m_input_image_size.width, _m_input_image_size.height);
    const float scale = static_cast<float>(input_node_h) / static_cast<float>(long_side);
    const cv::Size target_size(
        static_cast<int>(scale * static_cast<float>(_m_input_image_size.width)),
        static_cast<int>(scale * static_cast<float>(_m_input_image_size.height)));

    cv::Mat result;
    cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
    cv::resize(result, result, target_size);
    result.convertTo(result, CV_32FC3);
    cv::divide(result, 255.0, result);
    const auto pad_h = input_node_h - target_size.height;
    const auto pad_w = input_node_w - target_size.width;
    cv::copyMakeBorder(result, result, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, 0.0);

    const auto chw_data = CvUtils::convert_to_chw_vec(result);
    std::vector<NamedTensor> inputs;
    NamedTensor named;
    named.name = this->session().inputs().front().name;
    named.tensor = jinq::models::backend::Tensor::make<float>(
        {1, 3, input_node_h, input_node_w});
    if (chw_data.size() * sizeof(float) != named.tensor.byte_size()) {
        LOG(ERROR) << "preprocessed chw data mismatches the input tensor";
        return {};
    }
    std::memcpy(named.tensor.buffer.data(), chw_data.data(), named.tensor.byte_size());
    inputs.push_back(std::move(named));
    return inputs;
}

template <typename INPUT, typename OUTPUT>
cv::Mat FastSamSegmentor<INPUT, OUTPUT>::upscale_mask_image(const cv::Mat& mask) const {
    const auto input_node_h = _m_preds_mask_size.height;
    const auto input_node_w = _m_preds_mask_size.width;
    const auto long_side = std::max(_m_input_image_size.width, _m_input_image_size.height);
    const float scale = static_cast<float>(input_node_h) / static_cast<float>(long_side);
    const cv::Size target_size(
        static_cast<int>(scale * static_cast<float>(_m_input_image_size.width)),
        static_cast<int>(scale * static_cast<float>(_m_input_image_size.height)));
    const auto pad_h = input_node_h - target_size.height;
    const auto pad_w = input_node_w - target_size.width;

    cv::Mat result_mask;
    const cv::Rect src_mask_roi =
        cv::Rect(0, 0, mask.cols - pad_w, mask.rows - pad_h) & cv::Rect(0, 0, mask.cols, mask.rows);
    mask(src_mask_roi).copyTo(result_mask);
    cv::resize(result_mask, result_mask, _m_input_image_size, 0.0, 0.0, cv::INTER_LINEAR);
    return result_mask;
}

template <typename INPUT, typename OUTPUT>
StatusCode FastSamSegmentor<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor>& outputs,
                                                        OUTPUT& output) {
    const auto* output0 = find_output(outputs, "output0");
    const auto* output1 = find_output(outputs, "output1");
    if (output0 == nullptr || output1 == nullptr) {
        LOG(ERROR) << "fastsam outputs 'output0'/'output1' missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto& preds = output0->tensor;
    const auto& protos = output1->tensor;
    if (preds.shape.size() != 3 || protos.shape.size() != 4) {
        LOG(ERROR) << "unexpected fastsam output shapes: "
                   << jinq::models::backend::shape_to_string(preds.shape) << " / "
                   << jinq::models::backend::shape_to_string(protos.shape);
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto* preds_data = preds.template data<float>();
    const auto bbox_info_len = preds.shape[1];
    const auto bbox_nums = preds.shape[2];

    // transpose [info, N] into per-box rows
    std::vector<std::vector<float>> total_preds(bbox_nums,
                                                std::vector<float>(static_cast<size_t>(bbox_info_len), 0.0f));
    for (int64_t info_idx = 0; info_idx < bbox_info_len; ++info_idx) {
        for (int64_t box_idx = 0; box_idx < bbox_nums; ++box_idx) {
            total_preds[static_cast<size_t>(box_idx)][static_cast<size_t>(info_idx)] =
                preds_data[info_idx * bbox_nums + box_idx];
        }
    }

    std::vector<PredsBBox> threshed_preds;
    for (const auto& bbox : total_preds) {
        const auto conf = bbox[4];
        if (conf > _m_conf_thresh) {
            PredsBBox b;
            b.score = conf;
            b.masks = {bbox.begin() + 5, bbox.end()};
            auto cx = bbox[0];
            auto cy = bbox[1];
            auto width = bbox[2];
            auto height = bbox[3];
            auto x = std::max(cx - width / 2.0f, 0.0f);
            auto y = std::max(cy - height / 2.0f, 0.0f);
            b.bbox = cv::Rect2f(x, y, width, height);
            threshed_preds.push_back(b);
        }
    }
    auto nms_result = CvUtils::nms_bboxes(threshed_preds, _m_iou_thresh);

    // mask protos: [1,C,mh,mw] chw -> mat (mh*mw, C)
    const auto c = protos.shape[1];
    const auto mh = _m_preds_mask_size.height;
    const auto mw = _m_preds_mask_size.width;
    const auto* protos_data = protos.template data<float>();
    std::vector<float> protos_vec(protos_data, protos_data + protos.element_count());
    auto mask_proto_hwc = CvUtils::convert_to_hwc_vec(protos_vec, 1, static_cast<int>(c), mh * mw);
    // rows = c proto channels, cols = mh*mw mask positions
    cv::Mat mask_proto(cv::Size(mh * mw, static_cast<int>(c)), CV_32FC1, mask_proto_hwc.data());

    std::vector<cv::Mat> predicted_masks;
    const float downscale_h = static_cast<float>(mh) / static_cast<float>(_m_input_tensor_size.height);
    const float downscale_w = static_cast<float>(mw) / static_cast<float>(_m_input_tensor_size.width);
    for (const auto& bbox : nms_result) {
        cv::Mat mask_in(cv::Size(static_cast<int>(c), 1), CV_32FC1,
                        const_cast<float*>(bbox.masks.data()));
        cv::Mat mask_output = mask_in * mask_proto;
        mask_output = mask_output.reshape(1, {mw, mh});
        cv::Mat tmp_exp(mask_output.size(), CV_32FC1);
        cv::exp(-mask_output, tmp_exp);
        cv::Mat sigmoid_output = 1.0f / (1.0f + tmp_exp);

        // crop mask via downscaled bbox
        for (auto row = 0; row < sigmoid_output.rows; ++row) {
            auto* row_data = sigmoid_output.ptr<float>(row);
            for (auto col = 0; col < sigmoid_output.cols; ++col) {
                const auto scaled_bbox_tlx = static_cast<int>(bbox.bbox.x * downscale_w);
                const auto scaled_bbox_tly = static_cast<int>(bbox.bbox.y * downscale_h);
                const auto scaled_bbox_rbx = scaled_bbox_tlx + static_cast<int>(bbox.bbox.width * downscale_w);
                const auto scaled_bbox_rby = scaled_bbox_tly + static_cast<int>(bbox.bbox.height * downscale_h);
                if (!(row > scaled_bbox_tly && row < scaled_bbox_rby && col > scaled_bbox_tlx &&
                      col < scaled_bbox_rbx)) {
                    row_data[col] = 0.0f;
                }
            }
        }

        auto upscaled = upscale_mask_image(sigmoid_output);
        cv::Mat mask = cv::Mat::zeros(upscaled.size(), CV_8UC1);
        for (auto row = 0; row < upscaled.rows; ++row) {
            auto* row_data = mask.ptr(row);
            for (auto col = 0; col < upscaled.cols; ++col) {
                if (upscaled.template ptr<float>(row)[col] >= 0.5) {
                    row_data[col] = 255;
                }
            }
        }
        predicted_masks.push_back(std::move(mask));
    }

    // reorder by area and paint the everything mask with object ids
    auto comp_area = [](const cv::Mat& a, const cv::Mat& b) -> bool {
        return cv::countNonZero(a) >= cv::countNonZero(b);
    };
    std::sort(predicted_masks.begin(), predicted_masks.end(), comp_area);
    cv::Mat everything_mask = cv::Mat::zeros(_m_input_image_size, CV_32SC1);
    for (size_t idx = 0; idx < predicted_masks.size(); ++idx) {
        everything_mask.setTo(static_cast<int>(idx + 1), predicted_masks[idx]);
    }
    output = std::move(everything_mask);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
FastSamSegmentor<INPUT, OUTPUT>::FastSamSegmentor()
    : jinq::models::BackendCvModel<INPUT, OUTPUT>("FAST_SAM") {}

}
}
}
