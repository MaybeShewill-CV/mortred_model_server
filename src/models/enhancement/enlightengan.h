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

namespace jinq {
namespace models {
namespace enhancement {
using jinq::common::StatusCode;

template <typename INPUT, typename OUTPUT>
class EnlightenGan : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    EnlightenGan();
    ~EnlightenGan() override = default;

    EnlightenGan(const EnlightenGan& transformer) = delete;
    EnlightenGan& operator=(const EnlightenGan& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode on_init(const toml::table& params) override;

    cv::Size _m_input_size_user;
    cv::Size _m_input_size_host;
    cv::Mat _m_input_alpha;
};

} // namespace enhancement
} // namespace models
} // namespace jinq

#include "enlightengan.inl"

#endif // MM_AI_SERVER_ENLIGHTENGAN_H
