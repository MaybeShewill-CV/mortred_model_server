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
#include "server/matting/modnet_server.h"
#include "server/matting/pp_matting_server.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace matting {

using jinq::models::matting::ModNetMatting;
using jinq::models::matting::PPMatting;
using jinq::server::matting::ModNetServer;
using jinq::server::matting::PPMattingServer;

// create modnet matting model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_modnet_segmentor(const std::string& segmentor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<ModNetMatting<INPUT, OUTPUT> >(segmentor_name);
    return model_factory.create(segmentor_name);
}

// create pp human matting model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_ppmatting_segmentor(const std::string& segmentor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<PPMatting<INPUT, OUTPUT> >(segmentor_name);
    return model_factory.create(segmentor_name);
}

// create pp matting server
inline std::unique_ptr<BaseAiServer> create_pp_matting_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<PPMattingServer>(server_name);
    return server_factory.create(server_name);
}

// create modnet matting server
inline std::unique_ptr<BaseAiServer> create_modnet_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<ModNetServer>(server_name);
    return server_factory.create(server_name);
}

}  // namespace matting
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_MATTING_TASK_H
