/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: lightglue.h
 * Date: 23-11-3
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LIGHTGLUE_H
#define MORTRED_MODEL_SERVER_LIGHTGLUE_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/multi_session_model.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace feature_point {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT>
class LightGlue : public mortred::models::backend::MultiSessionModel<LightGlue<INPUT, OUTPUT>, INPUT, OUTPUT> {
    using Base = mortred::models::backend::MultiSessionModel<LightGlue<INPUT, OUTPUT>, INPUT, OUTPUT>;
    friend Base;

  public:
    LightGlue();
    ~LightGlue() override = default;

    LightGlue(const LightGlue &transformer) = delete;
    LightGlue &operator=(const LightGlue &transformer) = delete;

  protected:
    struct FeaturePoints {
        std::vector<float> keypoints;
        std::vector<float> normalized_keypoints;
        std::vector<float> descriptors;
    };

    StatusCode classify_extractor_outputs(const std::vector<mortred::models::backend::NamedTensor> &outputs) const;
    StatusCode classify_matcher_outputs(const std::vector<mortred::models::backend::NamedTensor> &outputs) const;
    StatusCode classify_feature_layout(const FeaturePoints &src_features, const FeaturePoints &dst_features) const;

  private:
    StatusCode on_init(const toml::table &params) override;

    StatusCode run_sessions(const INPUT &input, OUTPUT &output) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    cv::Mat preprocess_image(const cv::Mat &input_image) const;

    StatusCode extract_feature_points(const cv::Mat &input_image, FeaturePoints &feature_points) const;

    StatusCode match_feature_points(const FeaturePoints &src_features, const FeaturePoints &dst_features,
                                    mortred::models::io_define::feature_point::std_feature_point_match_output &matches) const;

    static std::vector<mortred::models::backend::SessionSpec> sessions();

    float _m_extract_score_threshold = 0.0f;
    float _m_match_score_threshold = 0.0f;
    float _m_long_side_length = 512.0f;
};

} // namespace feature_point
} // namespace models
} // namespace mortred

#include "lightglue.inl"

#endif // MORTRED_MODEL_SERVER_LIGHTGLUE_H
