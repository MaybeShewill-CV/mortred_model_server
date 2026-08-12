/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: Codex
 * File: sam_task.h
 * Date: 2026-08-12
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SAM_TASK_H
#define MORTRED_MODEL_SERVER_SAM_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/segment_anything/sam_prediction/sam_predictor.h"
#include "models/segment_anything/sam_automask_generator/sam_automask_generator.h"
#include "models/segment_anything/fast_sam/fast_sam_segmentor.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;

namespace segment_anything {

using jinq::models::segment_anything::SamPredictor;
using jinq::models::segment_anything::SamAutoMaskGenerator;
using jinq::models::segment_anything::FastSamSegmentor;

/***
 * create sam prompt predictor
 * @tparam INPUT
 * @tparam OUTPUT
 * @param model_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_sam_predictor(const std::string& model_name) {
    REGISTER_AI_MODEL(SamPredictor, model_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(model_name);
}

/***
 * create sam auto mask generator
 * @tparam INPUT
 * @tparam OUTPUT
 * @param model_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_sam_auto_mask_generator(const std::string& model_name) {
    REGISTER_AI_MODEL(SamAutoMaskGenerator, model_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(model_name);
}

/***
 * create fast sam segmentor
 * @tparam INPUT
 * @tparam OUTPUT
 * @param model_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_fast_sam_segmentor(const std::string& model_name) {
    REGISTER_AI_MODEL(FastSamSegmentor, model_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(model_name);
}

} // namespace segment_anything
} // namespace factory
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_SAM_TASK_H
