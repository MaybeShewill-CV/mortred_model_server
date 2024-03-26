/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: matting_task.h
* Date: 22-7-22
************************************************/

#ifndef MORTRED_MODEL_SERVER_MATTING_TASK_H
#define MORTRED_MODEL_SERVER_MATTING_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/matting/modnet_matting.h"
#include "models/matting/pp_matting.h"
#include "server/matting/pp_matting_server.h"
#include "server/matting/modnet_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace matting {
using jinq::models::matting::ModNetMatting;
using jinq::models::matting::PPMatting;

using jinq::server::matting::PPMattingServer;
using jinq::server::matting::ModNetServer;

/***
 * create modnet human matting instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_modnet_segmentor(const std::string& segmentor_name) {
    REGISTER_AI_MODEL(ModNetMatting, segmentor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(segmentor_name);
}

/***
 * create pp human matting instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_ppmatting_segmentor(const std::string& segmentor_name) {
    REGISTER_AI_MODEL(PPMatting, segmentor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(segmentor_name);
}

/***
 * create pp matting server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_pp_matting_server(const std::string& server_name) {
    REGISTER_AI_SERVER(PPMattingServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create modnet matting server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_modnet_server(const std::string& server_name) {
    REGISTER_AI_SERVER(ModNetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

}
}
}

#endif //MORTRED_MODEL_SERVER_MATTING_TASK_H
