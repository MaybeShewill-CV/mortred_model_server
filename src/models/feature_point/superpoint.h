/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: superpoint.h
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SUPERPOINT_H
#define MORTRED_MODEL_SERVER_SUPERPOINT_H

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace feature_point {

template<typename INPUT, typename OUTPUT>
class SuperPoint : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    SuperPoint();
    ~SuperPoint() override = default;

    SuperPoint(const SuperPoint& transformer) = delete;
    SuperPoint& operator=(const SuperPoint& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    jinq::common::StatusCode on_init(const toml::table& params) override;

    const jinq::models::backend::NamedTensor* find_output(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const std::string& name) const;

    void decode_fp_location_and_score(
        const jinq::models::backend::NamedTensor& semi,
        jinq::models::io_define::feature_point::std_feature_point_output& key_points) const;

    void decode_fp_descriptor(
        const jinq::models::backend::NamedTensor& desc,
        jinq::models::io_define::feature_point::std_feature_point_output& key_points) const;

    // score thresh
    double _m_score_threshold = 0.015;
    // nms thresh
    double _m_nms_threshold = 4.0;
    // dense map cell size
    int _m_cell_size = 8;
    // user image size of the current run
    cv::Size _m_input_size_user = cv::Size();
    // network input node size
    cv::Size _m_input_size_host = cv::Size();
};

}
}
}

#include "superpoint.inl"

#endif //MORTRED_MODEL_SERVER_SUPERPOINT_H
