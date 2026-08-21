/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: libface_detector.h
 * Date: 22-6-10
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H
#define MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H

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
class LibFaceDetector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    LibFaceDetector();
    ~LibFaceDetector() override = default;

    LibFaceDetector(const LibFaceDetector& transformer) = delete;
    LibFaceDetector& operator=(const LibFaceDetector& transformer) = delete;

  private:
    struct FaceAnchor {
        double cx = 0.0;
        double cy = 0.0;
        double s_kx = 0.0;
        double s_ky = 0.0;
    };

    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    jinq::common::StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    jinq::common::StatusCode on_init(const toml::table& params) override;

    const jinq::models::backend::NamedTensor* find_output(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const std::string& name) const;

    std::vector<FaceAnchor> generate_prior_anchors() const;

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

} // namespace object_detection
} // namespace models
} // namespace jinq

#include "libface_detector.inl"

#endif // MORTRED_MODEL_SERVER_LIBFACE_DETECTOR_H
