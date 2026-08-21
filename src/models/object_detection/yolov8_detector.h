/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov8_detector.h
* Date: 24-3-13
************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H

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
class YoloV8Detector : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    YoloV8Detector();
    ~YoloV8Detector() override = default;

    YoloV8Detector(const YoloV8Detector& transformer) = delete;
    YoloV8Detector& operator=(const YoloV8Detector& transformer) = delete;

  private:
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat& image) override;

    StatusCode postprocess(
        const std::vector<jinq::models::backend::NamedTensor>& outputs,
        OUTPUT& output) override;

    StatusCode on_init(const toml::table& params) override;

    cv::Rect2f transform_bboxes(const cv::Rect2d& bbox) const;

    // score thresh
    double _m_score_threshold = 0.4;
    // nms thresh
    double _m_nms_threshold = 0.35;
    // top_k keep thresh
    int _m_keep_topk = 250;
    // class nums
    int _m_class_nums = 80;
    // class id to names
    std::map<int, std::string> _m_class_id2names;
    // host input node size (network space)
    cv::Size _m_input_size_host = cv::Size();
    // user image size of the current run
    cv::Size _m_input_size_user = cv::Size();
};

}
}
}

#include "yolov8_detector.inl"

#endif //MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H
