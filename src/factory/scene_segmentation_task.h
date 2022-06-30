/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: scene_segmentation.h
* Date: 22-6-9
************************************************/

#ifndef MM_AI_SERVER_SCENE_SEGMENTATION_H
#define MM_AI_SERVER_SCENE_SEGMENTATION_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/scene_segmentation/bisenetv2.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;

namespace scene_segmentation {
using jinq::models::scene_segmentation::BiseNetV2;

/***
 * create bisenetv2 scene segmentation instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_bisenetv2_segmentor(const std::string& segmentor_name) {
    REGISTER_AI_MODEL(BiseNetV2, segmentor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(segmentor_name);
}

}
}
}

#endif //MM_AI_SERVER_SCENE_SEGMENTATION_H
