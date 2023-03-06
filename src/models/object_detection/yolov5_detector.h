/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: yolov5_detector.h
* Date: 22-6-7
************************************************/

#ifndef MM_AI_SERVER_YOLOV5DETECTOR_H
#define MM_AI_SERVER_YOLOV5DETECTOR_H

#include <memory>

#include "toml/toml.hpp"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class YoloV5Detector : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * constructor
    * @param config
    */
    YoloV5Detector();

    /***
     *
     */
    ~YoloV5Detector() override;

    /***
    * constructor
    * @param transformer
    */
    YoloV5Detector(const YoloV5Detector& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    YoloV5Detector& operator=(const YoloV5Detector& transformer) = delete;

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
     * if yolov5t detector successfully initialized
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

#include "yolov5_detector.inl"

#endif //MM_AI_SERVER_YOLOV5DETECTOR_H
