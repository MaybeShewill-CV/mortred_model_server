/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mobilenetv2.h
* Date: 22-6-13
************************************************/

#ifndef MORTRED_MODEL_SERVER_MOBILENETV2_H
#define MORTRED_MODEL_SERVER_MOBILENETV2_H

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
class MobileNetv2 : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    MobileNetv2();
    ~MobileNetv2() override = default;

    MobileNetv2(const MobileNetv2& transformer) = delete;
    MobileNetv2& operator=(const MobileNetv2& transformer) = delete;

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

#include "mobilenetv2.inl"

#endif //MORTRED_MODEL_SERVER_MOBILENETV2_H
