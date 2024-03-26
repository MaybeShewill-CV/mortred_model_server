/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: feature_point_task.h
 * Date: 22-6-15
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H
#define MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/feature_point/superpoint.h"
#include "server/feature_point/superpoint_fp_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace feature_point {
using jinq::models::feature_point::SuperPoint;

using jinq::server::feature_point::SuperpointFpServer;

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

/***
 * create superpoint feature point detection server
 * @param server_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_superpoint_fp_server(const std::string& server_name) {
    REGISTER_AI_SERVER(SuperpointFpServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

} // namespace feature_point
} // namespace factory
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H
