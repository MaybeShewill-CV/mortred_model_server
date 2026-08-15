/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: feature_point_task.h
* Date: 22-6-15
************************************************/

#ifndef MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H
#define MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/feature_point/superpoint.h"
#include "server/abstract_server.h"
#include "server/feature_point/superpoint_fp_server.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace feature_point {

using jinq::models::feature_point::SuperPoint;
using jinq::server::feature_point::SuperpointFpServer;

// create superpoint feature point extractor model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_superpoint_extractor(const std::string& extractor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<SuperPoint<INPUT, OUTPUT> >(extractor_name);
    return model_factory.create(extractor_name);
}

// create superpoint feature point server
inline std::unique_ptr<BaseAiServer> create_superpoint_fp_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<SuperpointFpServer>(server_name);
    return server_factory.create(server_name);
}

}  // namespace feature_point
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H
