/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: scene_segmentation.h
* Date: 22-6-9
************************************************/

#ifndef MM_AI_SERVER_SCENE_SEGMENTATION_TASK_H
#define MM_AI_SERVER_SCENE_SEGMENTATION_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
// model header
#include "models/scene_segmentation/bisenetv2.h"
#include "models/scene_segmentation/pp_humanseg.h"
#include "models/scene_segmentation/msocrnet.h"
#include "models/scene_segmentation/hrnet_segmentation.h"
// server header
#include "server/scene_segmentation/bisenetv2_server.h"
#include "server/scene_segmentation/pphuman_seg_server.h"
#include "server/scene_segmentation/hrnet_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace scene_segmentation {
using jinq::models::scene_segmentation::BiseNetV2;
using jinq::models::scene_segmentation::PPHumanSeg;
using jinq::models::scene_segmentation::MsOcrNet;
using jinq::models::scene_segmentation::HRNetSegmentation;

using jinq::server::scene_segmentation::BiseNetV2Server;
using jinq::server::scene_segmentation::PPHumanSegServer;
using jinq::server::scene_segmentation::HRNetServer;

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
 * create pp human seg instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_pphuman_segmentor(const std::string& segmentor_name) {
    REGISTER_AI_MODEL(PPHumanSeg, segmentor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(segmentor_name);
}

/***
 * create msocrnet scene segmentation instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_msocrnet_segmentor(const std::string& segmentor_name) {
    REGISTER_AI_MODEL(MsOcrNet, segmentor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(segmentor_name);
}

/***
 * create hrnet scene segmentation instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_hrnet_segmentor(const std::string& segmentor_name) {
    REGISTER_AI_MODEL(HRNetSegmentation, segmentor_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(segmentor_name);
}

/***
 * create bisenetv2 scene segmentation server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_bisenetv2_server(const std::string& server_name) {
    REGISTER_AI_SERVER(BiseNetV2Server, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create pphuman segmentation server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_pphuman_seg_server(const std::string& server_name) {
    REGISTER_AI_SERVER(PPHumanSegServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create hrnet scene segmentation server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_hrnet_server(const std::string& server_name) {
    REGISTER_AI_SERVER(HRNetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

}
}
}

#endif //MM_AI_SERVER_SCENE_SEGMENTATION_TASK_H
