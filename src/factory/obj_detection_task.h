/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: image_object_detection_task.h
* Date: 22-6-8
************************************************/

#ifndef MM_AI_SERVER_OBJ_DETECTION_TASK_H
#define MM_AI_SERVER_OBJ_DETECTION_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/object_detection/yolov5_detector.h"

namespace morted {
namespace factory {

using morted::factory::ModelFactory;
using morted::models::BaseAiModel;

namespace object_detection {
using morted::models::object_detection::YoloV5Detector;

/***
 * create yolov5 object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_yolov5_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(YoloV5Detector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(detector_name);
}

}
}
}

#endif //MM_AI_SERVER_OBJ_DETECTION_TASK_H
