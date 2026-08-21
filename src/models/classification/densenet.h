/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: densenet.h
* Date: 22-6-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_DENSENET_H
#define MORTRED_MODEL_SERVER_DENSENET_H

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
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
class DenseNet : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    DenseNet();
    ~DenseNet() override = default;

    DenseNet(const DenseNet& transformer) = delete;
    DenseNet& operator=(const DenseNet& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode on_init(const toml::table& params) override;

    // class id to names
    std::unordered_map<uint16_t, std::string> _m_class_id2names;
    // network input tensor size
    cv::Size _m_input_tensor_size = cv::Size(224, 224);
};

}
}
}

#include "densenet.inl"

#endif //MORTRED_MODEL_SERVER_DENSENET_H
