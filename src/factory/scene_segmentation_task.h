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
#include "models/scene_segmentation/modnet_matting.h"
#include "server/scene_segmentation//bisenetv2_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace scene_segmentation {
using jinq::models::scene_segmentation::BiseNetV2;
using jinq::models::scene_segmentation::ModNetMatting;

using jinq::server::scene_segmentation::BiseNetV2Server;

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
 * create libface detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_bisenetv2_server(const std::string& server_name) {
    REGISTER_AI_SERVER(BiseNetV2Server, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

}
}
}

#endif //MM_AI_SERVER_SCENE_SEGMENTATION_H
