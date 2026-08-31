/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: superpoint.inl
 * Date: 22-6-15
 ************************************************/

#include "superpoint.h"

#include "glog/logging.h"
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "models/backend/f32_output.h"
#include "models/backend/model_runtime.h"
#include "models/backend/request_geometry.h"

namespace jinq {
namespace models {
namespace feature_point {

using FeatureOutput = jinq::models::io_define::feature_point::std_feature_point_output;
using jinq::common::StatusCode;
using jinq::models::backend::NamedTensor;

template <typename INPUT, typename OUTPUT> StatusCode SuperPoint<INPUT, OUTPUT>::on_init(const toml::table &params) {
    if (params.contains("model_score_threshold")) {
        _m_score_threshold = params["model_score_threshold"].value_or<double>(0.0);
    }
    if (params.contains("model_nms_threshold")) {
        _m_nms_threshold = params["model_nms_threshold"].value_or<double>(0.0);
    }
    const auto &input_info = this->session().inputs().front();
    // dynamic batch (shape[0] == -1) is fine: spatial dims must be concrete
    if (input_info.shape.size() != 4 || input_info.shape[1] != 1 || input_info.shape[2] <= 0 || input_info.shape[3] <= 0) {
        LOG(ERROR) << "unexpected superpoint input shape: " << input_info.to_string() << ", expected static [N,1,H,W] (nchw)";
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_input_size_host.height = static_cast<int>(input_info.shape[2]);
    _m_input_size_host.width = static_cast<int>(input_info.shape[3]);
    if (_m_input_size_host.height % _m_cell_size != 0 || _m_input_size_host.width % _m_cell_size != 0) {
        LOG(ERROR) << "superpoint input size " << _m_input_size_host << " must be a multiple of the cell size " << _m_cell_size;
        return StatusCode::MODEL_INIT_FAILED;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
const NamedTensor *SuperPoint<INPUT, OUTPUT>::find_output(const std::vector<NamedTensor> &outputs, const std::string &name) const {
    for (const auto &item : outputs) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

template <typename INPUT, typename OUTPUT> std::vector<NamedTensor> SuperPoint<INPUT, OUTPUT>::preprocess(const cv::Mat &input_image) {
    // resize -> gray -> [0,1] (f32 nchw)
    auto result = jinq::models::backend::ImagePipeline(input_image)
                      .resize(_m_input_size_host)
                      .bgr_to_gray()
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
void SuperPoint<INPUT, OUTPUT>::decode_fp_location_and_score(const NamedTensor &semi, double score_threshold,
                                                             double nms_radius, FeatureOutput &key_points) const {
    const auto *host_data = semi.tensor.template data<float>();
    const int dense_map_row = _m_input_size_host.height / _m_cell_size;
    const int dense_map_col = _m_input_size_host.width / _m_cell_size;
    const int dense_map_channels = 65;

    // chw [65, r, c] -> planar [r, c, 65]
    std::vector<float> semi_tdata_reshape(semi.tensor.element_count(), 0.0f);
    for (auto row = 0; row < dense_map_row; ++row) {
        for (auto col = 0; col < dense_map_col; ++col) {
            for (auto channel = 0; channel < dense_map_channels; ++channel) {
                const auto to_index = row * dense_map_col * dense_map_channels + col * dense_map_channels + channel;
                const auto from_index = channel * dense_map_row * dense_map_col + row * dense_map_col + col;
                semi_tdata_reshape[to_index] = host_data[from_index];
            }
        }
    }
    cv::Mat dense(dense_map_row, dense_map_col, CV_32FC(dense_map_channels), semi_tdata_reshape.data());

    // softmax over the 65 channels, drop the dustbin channel
    std::vector<cv::Mat> dense_split;
    cv::split(dense, dense_split);
    cv::Mat dense_channel_sum = cv::Mat::zeros(dense_map_row, dense_map_col, CV_32FC1);
    for (auto &split : dense_split) {
        cv::exp(split, split);
        dense_channel_sum += split;
    }
    for (auto &split : dense_split) {
        cv::divide(split, dense_channel_sum, split);
    }
    cv::Mat dense_softmax;
    cv::merge(std::vector<cv::Mat>(dense_split.begin(), dense_split.end() - 1), dense_softmax);

    // select interest points
    for (auto row = 0; row < dense_map_row; ++row) {
        for (auto col = 0; col < dense_map_col; ++col) {
            for (int row_ext_index = 0; row_ext_index < _m_cell_size; ++row_ext_index) {
                for (int col_ext_index = 0; col_ext_index < _m_cell_size; ++col_ext_index) {
                    const int score_idx = row_ext_index * _m_cell_size + col_ext_index;
                    const float score = dense_softmax.at<cv::Vec<float, dense_map_channels - 1>>(row, col)[score_idx];
                    const int interest_pt_x = col * _m_cell_size + col_ext_index;
                    const int interest_pt_y = row * _m_cell_size + row_ext_index;
                    if (score >= score_threshold) {
                        jinq::models::io_define::feature_point::fp key_pt;
                        key_pt.location = cv::Point2f(static_cast<float>(interest_pt_x), static_cast<float>(interest_pt_y));
                        key_pt.score = score;
                        key_points.push_back(key_pt);
                    }
                }
            }
        }
    }

    // nms interest points
    std::sort(key_points.begin(), key_points.end(),
              [](const jinq::models::io_define::feature_point::fp &pt1, const jinq::models::io_define::feature_point::fp &pt2) {
                  return pt1.score >= pt2.score;
              });
    auto iter = key_points.begin();
    while (iter != key_points.end()) {
        auto comp = iter + 1;
        while (comp != key_points.end()) {
            const auto diff_x = iter->location.x - comp->location.x;
            const auto diff_y = iter->location.y - comp->location.y;
            const auto distance = std::sqrt(std::pow(diff_x, 2) + std::pow(diff_y, 2));
            if (distance <= nms_radius) {
                comp = key_points.erase(comp);
            } else {
                ++comp;
            }
        }
        ++iter;
    }
}

template <typename INPUT, typename OUTPUT>
void SuperPoint<INPUT, OUTPUT>::decode_fp_descriptor(const NamedTensor &desc, FeatureOutput &key_points) const {
    const auto *host_data = desc.tensor.template data<float>();
    const int desc_map_row = _m_input_size_host.height / _m_cell_size;
    const int desc_map_col = _m_input_size_host.width / _m_cell_size;
    const int desc_map_channels = 256;

    // chw [256, r, c] -> planar [r, c, 256]
    std::vector<float> desc_tdata_reshape(desc.tensor.element_count(), 0.0f);
    for (auto row = 0; row < desc_map_row; ++row) {
        for (auto col = 0; col < desc_map_col; ++col) {
            for (auto channel = 0; channel < desc_map_channels; ++channel) {
                const auto from_index = channel * desc_map_row * desc_map_col + row * desc_map_col + col;
                const auto to_index = row * desc_map_col * desc_map_channels + col * desc_map_channels + channel;
                desc_tdata_reshape[to_index] = host_data[from_index];
            }
        }
    }
    cv::Mat desc_map(desc_map_row, desc_map_col, CV_32FC(desc_map_channels), desc_tdata_reshape.data());

    // bilinear grid sample the descriptor at each keypoint
    for (auto &key_pt : key_points) {
        const float x = static_cast<float>(key_pt.location.x) / static_cast<float>(_m_cell_size);
        const float y = static_cast<float>(key_pt.location.y) / static_cast<float>(_m_cell_size);
        const float x1 = std::floor(x);
        const float x2 = std::ceil(x);
        const float y1 = std::floor(y);
        const float y2 = std::ceil(y);

        const auto f_q11 = desc_map.at<cv::Vec<float, 256>>(static_cast<int>(y1), static_cast<int>(x1));
        const auto f_q21 = desc_map.at<cv::Vec<float, 256>>(static_cast<int>(y1), static_cast<int>(x2));
        const auto f_q12 = desc_map.at<cv::Vec<float, 256>>(static_cast<int>(y2), static_cast<int>(x1));
        const auto f_q22 = desc_map.at<cv::Vec<float, 256>>(static_cast<int>(y2), static_cast<int>(x2));

        cv::Vec<float, 256> f_r1;
        cv::Vec<float, 256> f_r2;
        cv::Vec<float, 256> f_p;
        if (std::abs(x2 - x1) < 0.0000000001f) {
            f_r1 = f_q11;
            f_r2 = f_q11;
        } else {
            f_r1 = (x2 - x) / (x2 - x1) * f_q11 + (x - x1) / (x2 - x1) * f_q21;
            f_r2 = (x2 - x) / (x2 - x1) * f_q12 + (x - x1) / (x2 - x1) * f_q22;
        }
        if (std::abs(y2 - y1) < 0.00000000001f) {
            f_p = f_r1;
        } else {
            f_p = (y2 - y) / (y2 - y1) * f_r1 + (y - y1) / (y2 - y1) * f_r2;
        }
        key_pt.descriptor.resize(256);
        std::memcpy(key_pt.descriptor.data(), f_p.val, 256 * sizeof(float));
    }
}

template <typename INPUT, typename OUTPUT>
StatusCode SuperPoint<INPUT, OUTPUT>::postprocess(const std::vector<NamedTensor> &outputs,
                                                  const jinq::models::backend::InferenceContext &context, OUTPUT &output) {
    const auto *semi = find_output(outputs, "output_1");
    const auto *desc = find_output(outputs, "output_2");
    if (semi == nullptr || desc == nullptr) {
        LOG(ERROR) << "superpoint outputs 'output_1'/'output_2' missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    jinq::models::backend::GeometryScale geometry_scale;
    std::string geometry_error;
    if (!jinq::models::backend::make_geometry_scale(context, &geometry_scale, &geometry_error)) {
        LOG(ERROR) << "superpoint " << geometry_error;
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    const auto grid_height = context.network_size.height / _m_cell_size;
    const auto grid_width = context.network_size.width / _m_cell_size;
    auto semi_status = jinq::models::backend::validated_f32_named_output(
        outputs, "output_1", {jinq::models::backend::DType::F32, 4, {1, 65, grid_height, grid_width}}, "superpoint");
    if (semi_status != StatusCode::OK) {
        return semi_status;
    }
    const auto desc_status = jinq::models::backend::validated_f32_named_output(
        outputs, "output_2", {jinq::models::backend::DType::F32, 4, {1, 256, grid_height, grid_width}}, "superpoint");
    if (desc_status != StatusCode::OK) {
        return desc_status;
    }

    // request-level overrides (config TOML stays the default source); the
    // nms value is a pixel radius, deliberately named nms_radius in the spec
    double score_threshold = _m_score_threshold;
    double nms_radius = _m_nms_threshold;
    if (context.params != nullptr) {
        score_threshold = context.params->get_f32("score_threshold", static_cast<float>(score_threshold));
        nms_radius = context.params->get_i32("nms_radius", static_cast<int>(nms_radius));
    }

    FeatureOutput internal_out;
    decode_fp_location_and_score(*semi, score_threshold, nms_radius, internal_out);
    decode_fp_descriptor(*desc, internal_out);

    // rescale feature point locations into the user image space
    for (auto &pt : internal_out) {
        pt.location = jinq::models::backend::scale_point(pt.location, geometry_scale);
    }
    output = std::move(internal_out);
    return StatusCode::OK;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
SuperPoint<INPUT, OUTPUT>::SuperPoint() : jinq::models::BackendCvModel<INPUT, OUTPUT>("SUPERPOINT") {}

} // namespace feature_point
} // namespace models
} // namespace jinq
