/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: sam_task.h
* Date: 22-6-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_TASK_H
#define MORTRED_MODEL_SERVER_SAM_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/segment_anything/fast_sam/fast_sam_segmentor.h"
#include "models/segment_anything/sam_automask_generator/sam_automask_generator.h"
#include "models/segment_anything/sam_prediction/sam_predictor.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;

namespace segment_anything {

using jinq::models::segment_anything::FastSamSegmentor;
using jinq::models::segment_anything::SamAutoMaskGenerator;
using jinq::models::segment_anything::SamPredictor;

// create sam prompt predictor model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_sam_predictor(const std::string& model_name) {
    // 直接构造：模型创建不写全局注册表（无副作用、无互斥开销），
    // 消除"每次 create 都 register"反模式；name 仅保留以兼容调用方
    (void)model_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new SamPredictor<INPUT, OUTPUT>());
}

// create sam auto mask generator model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_sam_auto_mask_generator(const std::string& model_name) {
    // 直接构造：模型创建不写全局注册表（无副作用、无互斥开销），
    // 消除"每次 create 都 register"反模式；name 仅保留以兼容调用方
    (void)model_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new SamAutoMaskGenerator<INPUT, OUTPUT>());
}

// create fast sam segmentation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_fast_sam_segmentor(const std::string& model_name) {
    // 直接构造：模型创建不写全局注册表（无副作用、无互斥开销），
    // 消除"每次 create 都 register"反模式；name 仅保留以兼容调用方
    (void)model_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new FastSamSegmentor<INPUT, OUTPUT>());
}

}  // namespace segment_anything
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_SAM_TASK_H
