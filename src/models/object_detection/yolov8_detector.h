/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov8_detector.h
 * Date: 24-3-13
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class YoloV8Detector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:

    /***
    * constructor
     */
    YoloV8Detector();

    /***
     *
     */
    ~YoloV8Detector() override;

    /***
    * constructor
    * @param transformer
     */
    YoloV8Detector(const YoloV8Detector& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV8Detector& operator=(const YoloV8Detector& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode run(const INPUT& input, OUTPUT& output) override;


    /***
     * if yolov7 detector successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}

#include "yolov8_detector.inl"

#endif // MORTRED_MODEL_SERVER_YOLOV8_DETECTOR_H
