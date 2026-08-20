/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: centerface_detector.h
* Date: 23-10-18
************************************************/

#ifndef MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H
#define MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H

#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class CenterFaceDetector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    CenterFaceDetector();
    ~CenterFaceDetector() override = default;

    CenterFaceDetector(const CenterFaceDetector& transformer) = delete;
    CenterFaceDetector& operator=(const CenterFaceDetector& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    jinq::common::StatusCode on_init(const toml::table& params) override;

    const jinq::models::backend::NamedTensor* find_output(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const std::string& name) const;

    // score thresh
    double _m_score_threshold = 0.6;
    // nms thresh
    double _m_nms_threshold = 0.3;
    // top_k keep
    size_t _m_keep_topk = 250;
    // input image size
    cv::Size _m_input_size_user = cv::Size();
    // input node size
    cv::Size _m_input_size_host = cv::Size();
};

}
}
}

#include "centerface_detector.inl"

#endif //MORTRED_MODEL_SERVER_CENTERFACE_DETECTOR_H
