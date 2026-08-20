/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: hrnet_segmentation.h
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_HRNET_SEGMENTATION_H
#define MORTRED_MODEL_SERVER_HRNET_SEGMENTATION_H

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

template<typename INPUT, typename OUTPUT>
class HRNetSegmentation : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    HRNetSegmentation();
    ~HRNetSegmentation() override = default;

    HRNetSegmentation(const HRNetSegmentation& transformer) = delete;
    HRNetSegmentation& operator=(const HRNetSegmentation& transformer) = delete;

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

#include "hrnet_segmentation.inl"

#endif //MORTRED_MODEL_SERVER_HRNET_SEGMENTATION_H
