/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: dinov2.h
************************************************/

#ifndef MORTRED_MODEL_SERVER_DINOV2_H
#define MORTRED_MODEL_SERVER_DINOV2_H

#include <string>
#include <unordered_map>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace classification {

template<typename INPUT, typename OUTPUT>
class Dinov2 : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    Dinov2();
    ~Dinov2() override = default;

    Dinov2(const Dinov2& transformer) = delete;
    Dinov2& operator=(const Dinov2& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    jinq::common::StatusCode on_init(const toml::table& params) override;

    // class id to names
    std::unordered_map<uint16_t, std::string> _m_class_id2names;
    // network input tensor size
    cv::Size _m_input_tensor_size = cv::Size(224, 224);
};

}
}
}

#include "dinov2.inl"

#endif //MORTRED_MODEL_SERVER_DINOV2_H
