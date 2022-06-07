/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_factory.h
* Date: 22-6-7
************************************************/

#ifndef MM_AI_SERVER_BASE_FACTORY_H
#define MM_AI_SERVER_BASE_FACTORY_H

#include "models/base_model.h"
#include "models/image_ocr/db_text_detector.h"
#include "models/image_object_detection/yolov5_detector.h"

namespace morted {
namespace factory {

template<typename INPUT, typename OUTPUT>
class AiModelFactory {
public:
    /***
     *
     * @return
     */
    virtual morted::models::BaseAiModel<INPUT, OUTPUT>* create_model() = 0;

    virtual ~AiModelFactory() = default;
};

template<typename INPUT, typename OUTPUT>
class DBTextModelFactory : public AiModelFactory<INPUT, OUTPUT> {
public:
    morted::models::image_ocr::DBTextDetector<INPUT, OUTPUT>* create_model() override {
        return new morted::models::image_ocr::DBTextDetector<INPUT, OUTPUT>();
    }

    morted::models::image_ocr::DBTextDetector<INPUT, OUTPUT> create_model_object() {
        return morted::models::image_ocr::DBTextDetector<INPUT, OUTPUT>();
    }
};

template<typename INPUT, typename OUTPUT>
class Yolov5ModelFactory : public AiModelFactory<INPUT, OUTPUT> {
public:
    morted::models::image_object_detection::YoloV5Detector<INPUT, OUTPUT>* create_model() override {
        return new morted::models::image_object_detection::YoloV5Detector<INPUT, OUTPUT>();
    }
};
}
}


#endif //MM_AI_SERVER_BASE_FACTORY_H
