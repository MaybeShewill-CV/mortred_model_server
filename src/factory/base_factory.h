/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_factory.h
* Date: 22-6-7
************************************************/

#ifndef MM_AI_SERVER_BASE_FACTORY_H
#define MM_AI_SERVER_BASE_FACTORY_H

#include <memory>

#include "models/base_model.h"
#include "models/image_ocr/db_text_detector.h"
#include "models/image_object_detection/yolov5_detector.h"

namespace morted {
namespace factory {

using morted::models::BaseAiModel;
using morted::models::image_ocr::DBTextDetector;
using morted::models::image_object_detection::YoloV5Detector;

template<typename INPUT, typename OUTPUT>
class AiModelFactory {
public:
    /***
     *
     * @return
     */
    virtual std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_model() = 0;

    virtual ~AiModelFactory() = default;
};

template<typename INPUT, typename OUTPUT>
class DBTextModelFactory : public AiModelFactory<INPUT, OUTPUT> {
public:
    std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_model() override {
        return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new DBTextDetector<INPUT, OUTPUT>());
    }

    static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > static_create_model() {
        return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new DBTextDetector<INPUT, OUTPUT>());
    }
};

template<typename INPUT, typename OUTPUT>
class Yolov5ModelFactory : public AiModelFactory<INPUT, OUTPUT> {
public:
    std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_model() override {
        return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new YoloV5Detector<INPUT, OUTPUT>());
    }

    static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > static_create_model() {
        return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new YoloV5Detector<INPUT, OUTPUT>());
    }
};
}
}


#endif //MM_AI_SERVER_BASE_FACTORY_H
