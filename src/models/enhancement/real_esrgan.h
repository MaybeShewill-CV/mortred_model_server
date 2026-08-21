/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: real_esrgan.h
* Date: 22-9-29
************************************************/

#ifndef MORTRED_MODEL_SERVER_REALESRGAN_H
#define MORTRED_MODEL_SERVER_REALESRGAN_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace enhancement {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT>
class RealEsrGan : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    RealEsrGan();
    ~RealEsrGan() override = default;

    RealEsrGan(const RealEsrGan& transformer) = delete;
    RealEsrGan& operator=(const RealEsrGan& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode on_init(const toml::table& params) override;

    cv::Size _m_input_size_host;
};

} // namespace enhancement
} // namespace models
} // namespace jinq

#include "real_esrgan.inl"

#endif // MORTRED_MODEL_SERVER_REALESRGAN_H
