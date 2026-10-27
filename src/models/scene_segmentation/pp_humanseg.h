/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: pp_humanseg.h
 * Date: 22-7-20
 ************************************************/

#ifndef MM_AI_SERVER_PP_HUMANSEG_H
#define MM_AI_SERVER_PP_HUMANSEG_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace scene_segmentation {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class PPHumanSeg : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    PPHumanSeg();
    ~PPHumanSeg() override = default;

    PPHumanSeg(const PPHumanSeg &transformer) = delete;
    PPHumanSeg &operator=(const PPHumanSeg &transformer) = delete;

  protected:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    cv::Size _m_input_size_host = cv::Size();
};

} // namespace scene_segmentation
} // namespace models
} // namespace mortred

#include "pp_humanseg.inl"

#endif // MM_AI_SERVER_PP_HUMANSEG_H
