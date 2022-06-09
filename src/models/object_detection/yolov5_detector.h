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

namespace morted {
namespace models {
namespace object_detection {

template<typename INPUT, typename OUTPUT>
class YoloV5Detector : public morted::models::BaseAiModel<INPUT, OUTPUT> {
public:

    /***
    * 构造函数
    * @param config
    */
    YoloV5Detector();

    /***
     *
     */
    ~YoloV5Detector() override;

    /***
    * 赋值构造函数
    * @param transformer
    */
    YoloV5Detector(const YoloV5Detector& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    YoloV5Detector& operator=(const YoloV5Detector& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    morted::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    morted::common::StatusCode run(const INPUT& input, std::vector<OUTPUT>& output) override;


    /***
     * if db text detector successfully initialized
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
