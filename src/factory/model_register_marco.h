/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: model_register_marco.h
* Date: 22-6-8
************************************************/

#ifndef MM_AI_SERVER_MODEL_REGISTER_MARCO_H
#define MM_AI_SERVER_MODEL_REGISTER_MARCO_H

#include "factory/base_factory.h"
#include "models/model_io_define.h"
#include "models/image_ocr/db_text_detector.h"

using morted::models::BaseAiModel;
using morted::models::image_ocr::DBTextDetector;
using morted::factory::ModelRegistrar;

#define REGISTER_AI_MODEL(MODEL, MODEL_NAME, INPUT, OUTPUT) \
ModelRegistrar<BaseAiModel<INPUT, OUTPUT>, MODEL<INPUT, OUTPUT> > __xxx_##MODEL_(INPUT)_(OUTPUT)((MODEL_NAME));

#define LIST_ALL_REGISTERED_AI_MODELS(INPUT, OUTPUT) \
ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().list_registered_models();

#endif //MM_AI_SERVER_MODEL_REGISTER_MARCO_H
