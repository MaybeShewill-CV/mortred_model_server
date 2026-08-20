/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: depth_anything.h
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DEPTH_ANYTHING_H
#define MORTRED_MODEL_SERVER_DEPTH_ANYTHING_H

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace mono_depth_estimation {

template<typename INPUT, typename OUTPUT>
class DepthAnything : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    DepthAnything();
    ~DepthAnything() override = default;

    DepthAnything(const DepthAnything& transformer) = delete;
    DepthAnything& operator=(const DepthAnything& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    jinq::common::StatusCode on_init(const toml::table& params) override;

    // user image size of the current run
    cv::Size _m_input_size_user = cv::Size();
    // network input node size
    cv::Size _m_input_size_host = cv::Size();
};

}
}
}

#include "depth_anything.inl"

#endif //MORTRED_MODEL_SERVER_DEPTH_ANYTHING_H
