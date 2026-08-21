/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: clip_task.h
* Date: 22-6-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_CLIP_TASK_H
#define MORTRED_MODEL_SERVER_CLIP_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/clip/openai_clip.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;

namespace clip {

using jinq::models::clip::OpenAiClip;

// create openai clip model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_openai_clip(const std::string& model_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)model_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new OpenAiClip<INPUT, OUTPUT>());
}

}  // namespace clip
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_CLIP_TASK_H
