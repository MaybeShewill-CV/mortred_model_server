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
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<SamPredictor<INPUT, OUTPUT> >(model_name);
    return model_factory.create(model_name);
}

// create sam auto mask generator model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_sam_auto_mask_generator(const std::string& model_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<SamAutoMaskGenerator<INPUT, OUTPUT> >(model_name);
    return model_factory.create(model_name);
}

// create fast sam segmentation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_fast_sam_segmentor(const std::string& model_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<FastSamSegmentor<INPUT, OUTPUT> >(model_name);
    return model_factory.create(model_name);
}

}  // namespace segment_anything
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_SAM_TASK_H
