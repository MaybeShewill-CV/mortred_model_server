/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mono_depth_estimate_task.h
 * Date: 23-10-27
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H
#define MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/mono_depth_estimation/metric3d.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace mono_depth_estimation {
using jinq::models::mono_depth_estimation::Metric3D;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param estimator_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_metric3d_estimator(const std::string& estimator_name) {
    REGISTER_AI_MODEL(Metric3D, estimator_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(estimator_name);
}

} // namespace mono_depth_estimation
} // namespace factory
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H
