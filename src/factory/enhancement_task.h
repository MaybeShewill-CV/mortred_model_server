/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: enhancement_task.h
* Date: 22-6-13
************************************************/

#ifndef MM_AI_SERVER_ENHANCEMENT_TASK_H
#define MM_AI_SERVER_ENHANCEMENT_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/enhancement/enlightengan.h"

namespace morted {
namespace factory {

using morted::factory::ModelFactory;
using morted::models::BaseAiModel;

namespace enhancement {
using morted::models::enhancement::EnlightenGan;

/***
 * create enlighten=gan low light image enhancement
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_enlightengan_enhancementor(
    const std::string& enhancementor_name) {
    REGISTER_AI_MODEL(EnlightenGan, enhancementor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(enhancementor_name);
}

}
}
}

#endif //MM_AI_SERVER_ENHANCEMENT_TASK_H
