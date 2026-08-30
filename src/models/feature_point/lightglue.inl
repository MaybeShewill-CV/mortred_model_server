/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: lightglue.inl
 * Date: 23-11-3
 ************************************************/

#include "lightglue.h"

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "glog/logging.h"
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "models/backend/model_runtime.h"

namespace jinq {
namespace models {
namespace feature_point {

using FeatureMatchOutput = jinq::models::io_define::feature_point::std_feature_point_match_output;
using jinq::common::StatusCode;
using jinq::models::backend::InferenceSession;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorInfo;

template <typename INPUT, typename OUTPUT> std::vector<jinq::models::backend::SessionSpec> LightGlue<INPUT, OUTPUT>::sessions() {
    return {
        {"extractor", "extractor_backend", jinq::models::backend::IoSpec::input("image").f32().rank(4),
         jinq::models::backend::IoSpec::output("keypoints").i32().rank(3)},
        {"matcher", "matcher_backend", jinq::models::backend::IoSpec::input("kpts0").f32().rank(3),
         jinq::models::backend::IoSpec::output("matches0").i32().rank(2)},
    };
}

template <typename INPUT, typename OUTPUT> StatusCode LightGlue<INPUT, OUTPUT>::on_init(const toml::table &params) {
    if (params.contains("extract_score_thresh")) {
        _m_extract_score_threshold = static_cast<float>(params["extract_score_thresh"].value_or<double>(0.0));
    }
    if (params.contains("match_score_thresh")) {
        _m_match_score_threshold = static_cast<float>(params["match_score_thresh"].value_or<double>(0.0));
    }
    if (params.contains("long_side_length")) {
        _m_long_side_length = static_cast<float>(params["long_side_length"].value_or<double>(0.0));
    }
    if (_m_long_side_length <= 0.0f) {
        LOG(ERROR) << "lightglue long_side_length must be positive";
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto session_status = this->init_sessions();
    if (session_status != StatusCode::OK) {
        return session_status;
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
const NamedTensor *LightGlue<INPUT, OUTPUT>::find_output(const std::vector<NamedTensor> &outputs, const std::string &name) const {
    for (const auto &item : outputs) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

template <typename INPUT, typename OUTPUT> cv::Mat LightGlue<INPUT, OUTPUT>::preprocess_image(const cv::Mat &input_image) const {
    if (input_image.empty() || input_image.channels() != 3) {
        LOG(ERROR) << "lightglue input must be a non-empty 3-channel image";
        return {};
    }

    const auto long_side = std::max(input_image.cols, input_image.rows);
    const auto resize_scale = _m_long_side_length / static_cast<float>(long_side);
    const auto resize_height = static_cast<int>(static_cast<float>(input_image.rows) * resize_scale);
    const auto resize_width = static_cast<int>(static_cast<float>(input_image.cols) * resize_scale);
    if (resize_height <= 0 || resize_width <= 0) {
        LOG(ERROR) << "lightglue resized image size is empty: " << resize_width << "x" << resize_height;
        return {};
    }

    cv::Mat tmp;
    cv::resize(input_image, tmp, cv::Size(resize_width, resize_height), 0.0, 0.0, cv::INTER_AREA);
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
    if (tmp.type() != CV_32FC1) {
        tmp.convertTo(tmp, CV_32FC1);
    }
    tmp /= 255.0f;
    return tmp;
}

template <typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::extract_feature_points(const cv::Mat &input_image, FeaturePoints &feature_points) const {
    if (input_image.empty() || input_image.type() != CV_32FC1) {
        LOG(ERROR) << "lightglue extractor input must be a non-empty f32 gray image";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }

    const auto chw_data = jinq::common::CvUtils::convert_to_chw_vec(input_image);
    NamedTensor image;
    image.name = "image";
    image.tensor = Tensor::make<float>({1, 1, input_image.rows, input_image.cols});
    if (chw_data.size() * sizeof(float) != image.tensor.byte_size()) {
        LOG(ERROR) << "lightglue extractor image buffer size mismatch";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
    std::memcpy(image.tensor.buffer.data(), chw_data.data(), image.tensor.byte_size());

    std::vector<NamedTensor> inputs;
    inputs.push_back(std::move(image));
    std::vector<NamedTensor> outputs;
    const auto run_status = this->session("extractor")->run(inputs, outputs);
    if (run_status != StatusCode::OK) {
        return run_status;
    }

    const auto *keypoints = find_output(outputs, "keypoints");
    const auto *scores = find_output(outputs, "scores");
    const auto *descriptors = find_output(outputs, "descriptors");
    if (keypoints == nullptr || scores == nullptr || descriptors == nullptr) {
        LOG(ERROR) << "lightglue extractor outputs keypoints/scores/descriptors missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    if (keypoints->tensor.dtype != jinq::models::backend::DType::I32 || scores->tensor.dtype != jinq::models::backend::DType::F32 ||
        descriptors->tensor.dtype != jinq::models::backend::DType::F32) {
        LOG(ERROR) << "lightglue extractor output dtypes are unexpected";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto score_count = static_cast<size_t>(scores->tensor.element_count());
    if (score_count == 0 || keypoints->tensor.element_count() != static_cast<int64_t>(score_count * 2) ||
        descriptors->tensor.element_count() != static_cast<int64_t>(score_count * 256)) {
        LOG(ERROR) << "lightglue extractor output sizes mismatch: scores " << score_count << ", keypoints "
                   << keypoints->tensor.element_count() << ", descriptors " << descriptors->tensor.element_count();
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto *score_data = scores->tensor.template data<float>();
    const auto *keypoint_data = keypoints->tensor.template data<int32_t>();
    const auto *descriptor_data = descriptors->tensor.template data<float>();
    feature_points = FeaturePoints{};
    const auto normalize_scale = std::max(input_image.cols, input_image.rows);
    // The matcher ONNX graph expects keypoints centered at the image midpoint
    // and normalized by the long side; the end-to-end export performs the
    // equivalent Sub/Div operations before its positional encoding.
    for (size_t idx = 0; idx < score_count; ++idx) {
        if (score_data[idx] < _m_extract_score_threshold) {
            continue;
        }
        feature_points.keypoints.push_back(static_cast<float>(keypoint_data[idx * 2]));
        feature_points.keypoints.push_back(static_cast<float>(keypoint_data[idx * 2 + 1]));
        feature_points.normalized_keypoints.push_back((static_cast<float>(keypoint_data[idx * 2]) - input_image.cols * 0.5f) /
                                                      static_cast<float>(normalize_scale));
        feature_points.normalized_keypoints.push_back((static_cast<float>(keypoint_data[idx * 2 + 1]) - input_image.rows * 0.5f) /
                                                      static_cast<float>(normalize_scale));
        feature_points.descriptors.insert(feature_points.descriptors.end(), descriptor_data + idx * 256, descriptor_data + (idx + 1) * 256);
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::match_feature_points(const FeaturePoints &src_features, const FeaturePoints &dst_features,
                                                          FeatureMatchOutput &matches) const {
    matches.clear();
    const auto src_count = src_features.keypoints.size() / 2;
    const auto dst_count = dst_features.keypoints.size() / 2;
    if (src_features.descriptors.size() != src_count * 256 || dst_features.descriptors.size() != dst_count * 256 ||
        src_features.normalized_keypoints.size() != src_features.keypoints.size() ||
        dst_features.normalized_keypoints.size() != dst_features.keypoints.size()) {
        LOG(ERROR) << "lightglue selected feature keypoints/descriptors mismatch";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    if (src_count == 0 || dst_count == 0) {
        return StatusCode::OK;
    }

    std::vector<NamedTensor> inputs;
    NamedTensor kpts0;
    kpts0.name = "kpts0";
    kpts0.tensor = Tensor::make<float>({1, static_cast<int64_t>(src_count), 2});
    std::memcpy(kpts0.tensor.buffer.data(), src_features.normalized_keypoints.data(), kpts0.tensor.byte_size());
    inputs.push_back(std::move(kpts0));

    NamedTensor kpts1;
    kpts1.name = "kpts1";
    kpts1.tensor = Tensor::make<float>({1, static_cast<int64_t>(dst_count), 2});
    std::memcpy(kpts1.tensor.buffer.data(), dst_features.normalized_keypoints.data(), kpts1.tensor.byte_size());
    inputs.push_back(std::move(kpts1));

    NamedTensor desc0;
    desc0.name = "desc0";
    desc0.tensor = Tensor::make<float>({1, static_cast<int64_t>(src_count), 256});
    std::memcpy(desc0.tensor.buffer.data(), src_features.descriptors.data(), desc0.tensor.byte_size());
    inputs.push_back(std::move(desc0));

    NamedTensor desc1;
    desc1.name = "desc1";
    desc1.tensor = Tensor::make<float>({1, static_cast<int64_t>(dst_count), 256});
    std::memcpy(desc1.tensor.buffer.data(), dst_features.descriptors.data(), desc1.tensor.byte_size());
    inputs.push_back(std::move(desc1));

    std::vector<NamedTensor> outputs;
    const auto run_status = this->session("matcher")->run(inputs, outputs);
    if (run_status != StatusCode::OK) {
        return run_status;
    }

    const auto *match_indices = find_output(outputs, "matches0");
    const auto *match_scores = find_output(outputs, "mscores0");
    if (match_indices == nullptr || match_scores == nullptr) {
        LOG(ERROR) << "lightglue matcher outputs matches0/mscores0 missing";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    if (match_indices->tensor.dtype != jinq::models::backend::DType::I32 ||
        match_scores->tensor.dtype != jinq::models::backend::DType::F32) {
        LOG(ERROR) << "lightglue matcher output dtypes are unexpected";
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }
    const auto score_count = static_cast<size_t>(match_scores->tensor.element_count());
    if (match_indices->tensor.element_count() != static_cast<int64_t>(score_count * 2)) {
        LOG(ERROR) << "lightglue matcher output sizes mismatch: matches " << match_indices->tensor.element_count() << ", scores "
                   << score_count;
        return StatusCode::MODEL_EMPTY_OUTPUT;
    }

    const auto *index_data = match_indices->tensor.template data<int32_t>();
    const auto *score_data = match_scores->tensor.template data<float>();
    for (size_t idx = 0; idx < score_count; ++idx) {
        const auto match_score = score_data[idx];
        if (match_score < _m_match_score_threshold) {
            continue;
        }
        const auto src_index = index_data[idx * 2];
        const auto dst_index = index_data[idx * 2 + 1];
        if (src_index < 0 || static_cast<size_t>(src_index) >= src_count || dst_index < 0 || static_cast<size_t>(dst_index) >= dst_count) {
            continue;
        }

        namespace feature_point_io = jinq::models::io_define::feature_point;
        feature_point_io::fp src_point;
        src_point.location = cv::Point2f(static_cast<float>(src_features.keypoints[static_cast<size_t>(src_index) * 2]),
                                         static_cast<float>(src_features.keypoints[static_cast<size_t>(src_index) * 2 + 1]));
        feature_point_io::fp dst_point;
        dst_point.location = cv::Point2f(static_cast<float>(dst_features.keypoints[static_cast<size_t>(dst_index) * 2]),
                                         static_cast<float>(dst_features.keypoints[static_cast<size_t>(dst_index) * 2 + 1]));
        matches.push_back({std::make_pair(src_point, dst_point), match_score});
    }
    return StatusCode::OK;
}

template <typename INPUT, typename OUTPUT> StatusCode LightGlue<INPUT, OUTPUT>::run_sessions(const INPUT &input, OUTPUT &output) {
    if constexpr (std::is_same_v<INPUT, jinq::models::io_define::common_io::pair_mat_input>) {
        if (input.src_input_image.empty() || input.dst_input_image.empty()) {
            LOG(ERROR) << "lightglue source or destination input image is empty";
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }

        const auto src_size_user = input.src_input_image.size();
        const auto dst_size_user = input.dst_input_image.size();
        const auto src_preprocessed = preprocess_image(input.src_input_image);
        const auto dst_preprocessed = preprocess_image(input.dst_input_image);
        if (src_preprocessed.empty() || dst_preprocessed.empty()) {
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }

        FeaturePoints src_features;
        auto status = extract_feature_points(src_preprocessed, src_features);
        if (status != StatusCode::OK) {
            return status;
        }
        FeaturePoints dst_features;
        status = extract_feature_points(dst_preprocessed, dst_features);
        if (status != StatusCode::OK) {
            return status;
        }

        FeatureMatchOutput matches;
        status = match_feature_points(src_features, dst_features, matches);
        if (status != StatusCode::OK) {
            return status;
        }

        const auto src_width_scale = static_cast<float>(src_size_user.width) / static_cast<float>(src_preprocessed.cols);
        const auto src_height_scale = static_cast<float>(src_size_user.height) / static_cast<float>(src_preprocessed.rows);
        const auto dst_width_scale = static_cast<float>(dst_size_user.width) / static_cast<float>(dst_preprocessed.cols);
        const auto dst_height_scale = static_cast<float>(dst_size_user.height) / static_cast<float>(dst_preprocessed.rows);
        for (auto &matched_point : matches) {
            auto &src_point = matched_point.m_fp.first.location;
            src_point.x = (src_point.x + 0.5f) * src_width_scale - 0.5f;
            src_point.y = (src_point.y + 0.5f) * src_height_scale - 0.5f;
            auto &dst_point = matched_point.m_fp.second.location;
            dst_point.x = (dst_point.x + 0.5f) * dst_width_scale - 0.5f;
            dst_point.y = (dst_point.y + 0.5f) * dst_height_scale - 0.5f;
        }
        output = std::move(matches);
        return StatusCode::OK;
    } else {
        LOG(ERROR) << "lightglue expects pair_mat_input";
        return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
    }
}

template <typename INPUT, typename OUTPUT>
StatusCode LightGlue<INPUT, OUTPUT>::postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                                                 const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) {
    (void)outputs;
    (void)output;
    LOG(ERROR) << "lightglue is a multi-session model and must run through run_sessions";
    return StatusCode::MODEL_RUN_SESSION_FAILED;
}

/************* Export Function Sets *************/

template <typename INPUT, typename OUTPUT>
LightGlue<INPUT, OUTPUT>::LightGlue() : jinq::models::backend::MultiSessionModel<LightGlue<INPUT, OUTPUT>, INPUT, OUTPUT>("LIGHTGLUE") {}

} // namespace feature_point
} // namespace models
} // namespace jinq
