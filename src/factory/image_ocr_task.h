/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: image_ocr_task.h
* Date: 22-6-8
************************************************/

#ifndef MM_AI_SERVER_IMAGE_OCR_TASK_H
#define MM_AI_SERVER_IMAGE_OCR_TASK_H

#include "factory/base_factory.h"
#include "factory/model_register_marco.h"
#include "models/base_model.h"
#include "models/image_ocr/db_text_detector.h"
#include "models/image_object_detection/yolov5_detector.h"

namespace morted {
namespace factory {

using morted::factory::ModelFactory;
using morted::models::BaseAiModel;

namespace image_ocr {
using morted::models::image_ocr::DBTextDetector;

/***
 * create db text detector instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_dbtext_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(DBTextDetector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(detector_name);
}

}
}
}

#endif //MM_AI_SERVER_IMAGE_OCR_TASK_H
