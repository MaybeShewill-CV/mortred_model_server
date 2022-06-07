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

template<typename INPUT, typename OUTPUT>
class AiModelFactory {
public:
    /***
     *
     * @return
     */
    virtual std::unique_ptr<morted::models::BaseAiModel<INPUT, OUTPUT> > create_model() = 0;

    virtual ~AiModelFactory() = default;
};

template<typename INPUT, typename OUTPUT>
class DBTextModelFactory : public AiModelFactory<INPUT, OUTPUT> {
typedef morted::models::image_ocr::DBTextDetector<INPUT, OUTPUT> DBTextModel
public:
    std::unique_ptr<DBTextModel> create_model() override {
        return std::unique_ptr<DBTextModel>(new DBTextModel());
    }
};

template<typename INPUT, typename OUTPUT>
class Yolov5ModelFactory : public AiModelFactory<INPUT, OUTPUT> {
typedef morted::models::image_ocr::DBTextDetector<INPUT, OUTPUT> YoloV5Model
public:
    std::unique_ptr<YoloV5Model> create_model() override {
        return std::unique_ptr<DBTextModel>(new YoloV5Model());
    }
};
}
}


#endif //MM_AI_SERVER_BASE_FACTORY_H
