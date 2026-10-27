/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: db_text_detector.h
 * Date: 22-6-6
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_DB_TEXT_DETECTOR_H
#define MORTRED_MODEL_SERVER_DB_TEXT_DETECTOR_H

#include <string>
#include <vector>

#include "toml/toml.hpp"
#include <opencv2/opencv.hpp>

#include "models/backend/backend_cv_model.h"
#include "models/backend/request_geometry.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace mortred {
namespace models {
namespace ocr {
using mortred::common::StatusCode;

template <typename INPUT, typename OUTPUT> class DBTextDetector : public mortred::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    DBTextDetector();
    ~DBTextDetector() override = default;

    DBTextDetector(const DBTextDetector &transformer) = delete;
    DBTextDetector &operator=(const DBTextDetector &transformer) = delete;

  protected:
    std::vector<mortred::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    StatusCode postprocess(const std::vector<mortred::models::backend::NamedTensor> &outputs,
                           const mortred::models::backend::InferenceContext & /*context*/, OUTPUT &output) override;

    StatusCode on_init(const toml::table &params) override;

    StatusCode get_boxes_from_bitmap(const cv::Mat &seg_prob_mat, const cv::Mat &seg_score_mat,
                                    const mortred::models::backend::GeometryScale &geometry_scale,
                                    float score_threshold, int keep_topk, OUTPUT &output) const;

    // score thresh
    double _m_score_threshold = 0.4;
    // rotate bbox short side thresh
    float _m_sside_threshold = 3;
    // top_k keep thresh
    long _m_keep_topk = 250;
    cv::Size _m_input_size_host = cv::Size();
    // model io names
    std::string _m_input_name;
    std::string _m_output_name;
};

} // namespace ocr
} // namespace models
} // namespace mortred

#include "db_text_detector.inl"

#endif // MORTRED_MODEL_SERVER_DB_TEXT_DETECTOR_H
