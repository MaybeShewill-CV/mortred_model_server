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

namespace mortred {
namespace models {
namespace matting {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class ModNetMatting : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    ModNetMatting();
    ~ModNetMatting() override = default;

    ModNetMatting(const ModNetMatting &transformer) = delete;
    ModNetMatting &operator=(const ModNetMatting &transformer) = delete;

  protected:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    cv::Size _m_input_size_host = cv::Size();
};

} // namespace matting
} // namespace models
} // namespace mortred

#include "modnet_matting.inl"

#endif // MORTRED_MODEL_SERVER_MODNET_MATTING_H
