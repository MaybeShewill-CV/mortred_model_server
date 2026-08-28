/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: metric3d.h
 * Date: 23-10-27
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_METRIC3D_H
#define MORTRED_MODEL_SERVER_METRIC3D_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace mono_depth_estimation {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT> class Metric3D : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    Metric3D();
    ~Metric3D() override = default;

    Metric3D(const Metric3D &transformer) = delete;
    Metric3D &operator=(const Metric3D &transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    const jinq::models::backend::NamedTensor *find_output(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                                                          const std::string &name) const;

    void calculate_pad_info(int &pad_h, int &pad_w, const cv::Size &source_size) const;

    float calculate_label_scale_factor(const cv::Size &source_size) const;

    // focal length
    float _m_focal_length = 0.0f;
    // intrinsic params fx fy cx cy
    std::vector<float> _m_intrinsic_params = {0.0f, 0.0f, 0.0f, 0.0f};
    // network input node size
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace mono_depth_estimation
} // namespace models
} // namespace jinq

#include "metric3d.inl"

#endif // MORTRED_MODEL_SERVER_METRIC3D_H
