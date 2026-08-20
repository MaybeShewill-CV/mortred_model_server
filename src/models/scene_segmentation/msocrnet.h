/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: msocrnet.h
* Date: 23-3-11
************************************************/

#ifndef MM_AI_SERVER_MSOCRNET_H
#define MM_AI_SERVER_MSOCRNET_H

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace scene_segmentation {

template<typename INPUT, typename OUTPUT>
class MsOcrNet : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
public:
    MsOcrNet();
    ~MsOcrNet() override = default;

    MsOcrNet(const MsOcrNet& transformer) = delete;
    MsOcrNet& operator=(const MsOcrNet& transformer) = delete;

private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    jinq::common::StatusCode on_init(const toml::table& params) override;

    // network input size
    cv::Size _m_input_size_host = cv::Size();
    // user image size of the current run
    cv::Size _m_input_size_user = cv::Size();
    // input layout follows the backend (mnn: nhwc, onnx: nchw)
    bool _m_input_is_nhwc = false;
};

}
}
}

#include "msocrnet.inl"

#endif //MM_AI_SERVER_MSOCRNET_H
