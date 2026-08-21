/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: lightglue.h
* Date: 23-11-03
************************************************/

#ifndef MORTRED_MODEL_SERVER_LIGHTGLUE_H
#define MORTRED_MODEL_SERVER_LIGHTGLUE_H

#include <memory>
#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace feature_point {

template<typename INPUT, typename OUTPUT>
class LightGlue : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    LightGlue();
    ~LightGlue() override = default;

    LightGlue(const LightGlue& transformer) = delete;
    LightGlue& operator=(const LightGlue& transformer) = delete;

  private:
    struct FeaturePoints {
        std::vector<float> keypoints;
        std::vector<float> normalized_keypoints;
        std::vector<float> descriptors;
    };

    jinq::common::StatusCode on_init(const toml::table& params) override;

    jinq::common::StatusCode run_sessions(const INPUT& input, OUTPUT& output) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    jinq::common::StatusCode extract_feature_points(
        const cv::Mat& input_image, FeaturePoints& feature_points) const;

    jinq::common::StatusCode match_feature_points(
        const FeaturePoints& src_features,
        const FeaturePoints& dst_features,
        jinq::models::io_define::feature_point::std_feature_point_match_output& matches) const;

    const jinq::models::backend::NamedTensor* find_output(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const std::string& name) const;

    static const jinq::models::backend::TensorInfo* find_info(
        const jinq::models::backend::InferenceSession& session, const std::string& name);

    static jinq::common::StatusCode validate_io(
        const jinq::models::backend::InferenceSession& session);

    std::unique_ptr<jinq::models::backend::InferenceSession> _m_extractor;
    std::unique_ptr<jinq::models::backend::InferenceSession> _m_matcher;
    float _m_extract_score_threshold = 0.0f;
    float _m_match_score_threshold = 0.0f;
    float _m_long_side_length = 512.0f;
};

} // namespace feature_point
} // namespace models
} // namespace jinq

#include "lightglue.inl"

#endif // MORTRED_MODEL_SERVER_LIGHTGLUE_H
