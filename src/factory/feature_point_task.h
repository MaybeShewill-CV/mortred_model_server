/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: feature_point_task.h
 * Date: 22-6-15
 ************************************************/

#ifndef MM_AI_SERVER_FEATURE_POINT_TASK_H
#define MM_AI_SERVER_FEATURE_POINT_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/feature_point/superpoint.h"

namespace morted {
namespace factory {

using morted::factory::ModelFactory;
using morted::models::BaseAiModel;

namespace feature_point {
using morted::models::feature_point::SuperPoint;

/***
 * create superpoint image feature point task
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_superpoint_extractor(const std::string &extractor_name) {
    REGISTER_AI_MODEL(SuperPoint, extractor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(extractor_name);
}

} // namespace feature_point
} // namespace factory
} // namespace morted

#endif // MM_AI_SERVER_FEATURE_POINT_TASK_H
