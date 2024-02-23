/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov7_detector.h
* Date: 22-7-17
************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV7_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV7_DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class YoloV7Detector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    */
    YoloV7Detector();

    /***
     *
     */
    ~YoloV7Detector() override;

    /***
    * constructor
    * @param transformer
    */
    YoloV7Detector(const YoloV7Detector& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV7Detector& operator=(const YoloV7Detector& transformer) = delete;

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

#include "yolov7_detector.inl"

#endif //MORTRED_MODEL_SERVER_YOLOV7_DETECTOR_H
