/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov6_detector.h
* Date: 23-3-3
************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H

#include <map>
#include <string>
#include <vector>

#include "toml/toml.hpp"

#include "models/backend/backend_cv_model.h"
#include "models/backend/tensor.h"
#include "models/model_io_define.h"

namespace jinq {
namespace models {
namespace object_detection {
using jinq::common::StatusCode;

template<typename INPUT, typename OUTPUT>
class YoloV6Detector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    YoloV6Detector();
    ~YoloV6Detector() override = default;

    YoloV6Detector(const YoloV6Detector& transformer) = delete;
    YoloV6Detector& operator=(const YoloV6Detector& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode on_init(const toml::table& params) override;

    const jinq::models::backend::NamedTensor* find_output(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        const std::string& name) const;

    // score thresh
    double _m_score_threshold = 0.4;
    // nms thresh
    double _m_nms_threshold = 0.35;
    // top_k keep thresh
    long _m_keep_topk = 250;
    // class nums
    int _m_class_nums = 80;
    // class id to names
    std::map<int, std::string> _m_class_id2names;
    // input image size
    cv::Size _m_input_size_user = cv::Size();
    // input node size
    cv::Size _m_input_size_host = cv::Size();
};

}
}
}

#include "yolov6_detector.inl"

#endif //MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H
