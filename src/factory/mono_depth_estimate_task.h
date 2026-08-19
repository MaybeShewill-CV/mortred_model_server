/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mono_depth_estimate_task.h
* Date: 23-10-27
************************************************/

#ifndef MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H
#define MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/mono_depth_estimation/depth_anything.h"
#include "models/mono_depth_estimation/metric3d.h"
#include "server/abstract_server.h"
#include "server/generic_ai_server.h"
#include "server/response_serializers.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace mono_depth_estimation {

using jinq::models::mono_depth_estimation::DepthAnything;
using jinq::models::mono_depth_estimation::Metric3D;

// create metric3d mono depth estimation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_metric3d_estimator(const std::string& estimator_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<Metric3D<INPUT, OUTPUT> >(estimator_name);
    return model_factory.create(estimator_name);
}

// create depth anything mono depth estimation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_depth_anything_estimator(const std::string& estimator_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<DepthAnything<INPUT, OUTPUT> >(estimator_name);
    return model_factory.create(estimator_name);
}

// create metric3d depth estimation server
inline std::unique_ptr<BaseAiServer> create_metric3d_estimation_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::mono_depth_estimation::std_mde_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "METRIC3D_ESTIMATION_SERVER";
        spec.model_section = "METRIC3D";
        spec.display_name = "metric3d estimation";
        spec.make_worker = [](const std::string& name) {
            return create_metric3d_estimator<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_depth_estimation;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create depth anything depth estimation server
inline std::unique_ptr<BaseAiServer> create_depth_anything_estimation_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::mono_depth_estimation::std_mde_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "DEPTH_ANYTHING_ESTIMATION_SERVER";
        spec.model_section = "DEPTH_ANYTHING";
        spec.display_name = "depth anything estimation";
        spec.make_worker = [](const std::string& name) {
            return create_depth_anything_estimator<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_depth_estimation;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

}  // namespace mono_depth_estimation
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H
