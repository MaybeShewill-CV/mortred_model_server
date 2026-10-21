/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: enlightengan.h
 * Date: 22-6-13
 ************************************************/

#ifndef MM_AI_SERVER_ENLIGHTENGAN_H
#define MM_AI_SERVER_ENLIGHTENGAN_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace enhancement {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class EnlightenGan : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    EnlightenGan();
    ~EnlightenGan() override = default;

    EnlightenGan(const EnlightenGan &transformer) = delete;
    EnlightenGan &operator=(const EnlightenGan &transformer) = delete;

  protected:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;
};

} // namespace enhancement
} // namespace models
} // namespace mortred

#include "enlightengan.inl"

#endif // MM_AI_SERVER_ENLIGHTENGAN_H
