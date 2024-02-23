/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov6_detector.h
* Date: 23-3-3
************************************************/

#ifndef MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H
#define MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class YoloV6Detector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    * @param config
    */
    YoloV6Detector();

    /***
     *
     */
    ~YoloV6Detector() override;

    /***
    * constructor
    * @param transformer
    */
    YoloV6Detector(const YoloV6Detector& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV6Detector& operator=(const YoloV6Detector& transformer) = delete;

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
     * if yolov6 detector successfully initialized
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

#include "yolov6_detector.inl"

#endif //MORTRED_MODEL_SERVER_YOLOV6_DETECTOR_H
