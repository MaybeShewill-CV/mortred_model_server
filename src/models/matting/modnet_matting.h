/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: modnet_matting.h
 * Date: 22-7-19
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_MODNET_MATTING_H
#define MORTRED_MODEL_SERVER_MODNET_MATTING_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace matting {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT> class ModNetMatting : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    ModNetMatting();
    ~ModNetMatting() override = default;

    ModNetMatting(const ModNetMatting &transformer) = delete;
    ModNetMatting &operator=(const ModNetMatting &transformer) = delete;

  protected:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                           const jinq::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    // model input tensor size
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace matting
} // namespace models
} // namespace jinq

#include "modnet_matting.inl"

#endif // MORTRED_MODEL_SERVER_MODNET_MATTING_H
