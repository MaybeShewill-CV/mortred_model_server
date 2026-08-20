/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: pp_matting.h
* Date: 22-7-19
************************************************/

#ifndef MORTRED_MODEL_SERVER_PP_MATTING_H
#define MORTRED_MODEL_SERVER_PP_MATTING_H

#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace matting {

template<typename INPUT, typename OUTPUT>
class PPMatting : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    PPMatting();
    ~PPMatting() override = default;

    PPMatting(const PPMatting& transformer) = delete;
    PPMatting& operator=(const PPMatting& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    jinq::common::StatusCode on_init(const toml::table& params) override;

    // user input tensor size
    cv::Size _m_input_size_user = cv::Size();
    // model input tensor size
    cv::Size _m_input_size_host = cv::Size();
};

}
}
}

#include "pp_matting.inl"

#endif //MORTRED_MODEL_SERVER_PP_MATTING_H
