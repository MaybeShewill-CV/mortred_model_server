/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: hrnet_segmentation.h
 * Date: 23-11-17
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_HRNET_SEGMENTATION_H
#define MORTRED_MODEL_SERVER_HRNET_SEGMENTATION_H

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace scene_segmentation {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class HRNetSegmentation : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    HRNetSegmentation();
    ~HRNetSegmentation() override = default;

    HRNetSegmentation(const HRNetSegmentation &transformer) = delete;
    HRNetSegmentation &operator=(const HRNetSegmentation &transformer) = delete;

  protected:
    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

  private:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode on_init(const toml::table &params) override;

    // network input node size
    cv::Size _m_input_size_host = cv::Size();
};

} // namespace scene_segmentation
} // namespace models
} // namespace mortred

#include "hrnet_segmentation.inl"

#endif // MORTRED_MODEL_SERVER_HRNET_SEGMENTATION_H
