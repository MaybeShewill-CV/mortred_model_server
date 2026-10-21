/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: msocrnet.h
 * Date: 23-3-11
 ************************************************/

#ifndef MM_AI_SERVER_MSOCRNET_H
#define MM_AI_SERVER_MSOCRNET_H

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace scene_segmentation {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class MsOcrNet : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    MsOcrNet();
    ~MsOcrNet() override = default;

    MsOcrNet(const MsOcrNet &transformer) = delete;
    MsOcrNet &operator=(const MsOcrNet &transformer) = delete;

  protected:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    // network input size
    cv::Size _m_input_size_host = cv::Size();
    // input layout follows the backend (mnn: nhwc, onnx: nchw)
    bool _m_input_is_nhwc = false;
};

} // namespace scene_segmentation
} // namespace models
} // namespace mortred

#include "msocrnet.inl"

#endif // MM_AI_SERVER_MSOCRNET_H
