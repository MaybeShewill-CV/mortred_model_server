/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: matting_task.h
* Date: 22-7-22
************************************************/

#ifndef MORTRED_MODEL_SERVER_MATTING_TASK_H
#define MORTRED_MODEL_SERVER_MATTING_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/matting/modnet_matting.h"
#include "models/matting/pp_matting.h"
#include "server/abstract_server.h"
#include "server/generic_cv_server.h"
#include "server/response_serializers.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace matting {

using jinq::models::matting::ModNetMatting;
using jinq::models::matting::PPMatting;

// create modnet matting model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_modnet_segmentor(const std::string& segmentor_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)segmentor_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new ModNetMatting<INPUT, OUTPUT>());
}

// create pp human matting model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_ppmatting_segmentor(const std::string& segmentor_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)segmentor_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new PPMatting<INPUT, OUTPUT>());
}

// create pp matting server
inline std::unique_ptr<BaseAiServer> create_pp_matting_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::matting::std_matting_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "PP_MATTING_SERVER";
        spec.model_section = "PP_MATTING";
        spec.display_name = "pp matting";
        spec.make_worker = [](const std::string& name) {
            return create_ppmatting_segmentor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_matting;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create modnet matting server
inline std::unique_ptr<BaseAiServer> create_modnet_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::matting::std_matting_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "MODNET_SERVER";
        spec.model_section = "MODNET";
        spec.display_name = "modnet";
        spec.make_worker = [](const std::string& name) {
            return create_modnet_segmentor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_matting;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

}  // namespace matting
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_MATTING_TASK_H
