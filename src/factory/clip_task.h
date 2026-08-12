/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: Codex
 * File: clip_task.h
 * Date: 2026-08-12
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_CLIP_TASK_H
#define MORTRED_MODEL_SERVER_CLIP_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/clip/openai_clip.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;

namespace clip {

using jinq::models::clip::OpenAiClip;

/***
 * create openai clip model
 * @tparam INPUT
 * @tparam OUTPUT
 * @param model_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_openai_clip(const std::string& model_name) {
    REGISTER_AI_MODEL(OpenAiClip, model_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(model_name);
}

} // namespace clip
} // namespace factory
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_CLIP_TASK_H
