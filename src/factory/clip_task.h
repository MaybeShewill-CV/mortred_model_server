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
    // 直接构造：模型创建不写全局注册表（无副作用、无互斥开销），
    // 消除"每次 create 都 register"反模式；name 仅保留以兼容调用方
    (void)model_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new OpenAiClip<INPUT, OUTPUT>());
}

}  // namespace clip
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_CLIP_TASK_H
