/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: scene_segmentation.h
* Date: 22-6-9
************************************************/

#ifndef MORTRED_MODEL_SERVER_SCENE_SEGMENTATION_TASK_H
#define MORTRED_MODEL_SERVER_SCENE_SEGMENTATION_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/scene_segmentation/bisenetv2.h"
#include "models/scene_segmentation/hrnet_segmentation.h"
#include "models/scene_segmentation/msocrnet.h"
#include "models/scene_segmentation/pp_humanseg.h"
#include "server/abstract_server.h"
#include "server/scene_segmentation/bisenetv2_server.h"
#include "server/scene_segmentation/hrnet_server.h"
#include "server/scene_segmentation/pphuman_seg_server.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace scene_segmentation {

using jinq::models::scene_segmentation::BiseNetV2;
using jinq::models::scene_segmentation::HRNetSegmentation;
using jinq::models::scene_segmentation::MsOcrNet;
using jinq::models::scene_segmentation::PPHumanSeg;
using jinq::server::scene_segmentation::BiseNetV2Server;
using jinq::server::scene_segmentation::HRNetServer;
using jinq::server::scene_segmentation::PPHumanSegServer;

// create bisenetv2 scene segmentation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_bisenetv2_segmentor(const std::string& segmentor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<BiseNetV2<INPUT, OUTPUT> >(segmentor_name);
    return model_factory.create(segmentor_name);
}

// create pp human segmentation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_pphuman_segmentor(const std::string& segmentor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<PPHumanSeg<INPUT, OUTPUT> >(segmentor_name);
    return model_factory.create(segmentor_name);
}

// create msocrnet scene segmentation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_msocrnet_segmentor(const std::string& segmentor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<MsOcrNet<INPUT, OUTPUT> >(segmentor_name);
    return model_factory.create(segmentor_name);
}

// create hrnet scene segmentation model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_hrnet_segmentor(const std::string& segmentor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<HRNetSegmentation<INPUT, OUTPUT> >(segmentor_name);
    return model_factory.create(segmentor_name);
}

// create bisenetv2 scene segmentation server
inline std::unique_ptr<BaseAiServer> create_bisenetv2_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<BiseNetV2Server>(server_name);
    return server_factory.create(server_name);
}

// create pphuman segmentation server
inline std::unique_ptr<BaseAiServer> create_pphuman_seg_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<PPHumanSegServer>(server_name);
    return server_factory.create(server_name);
}

// create hrnet scene segmentation server
inline std::unique_ptr<BaseAiServer> create_hrnet_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<HRNetServer>(server_name);
    return server_factory.create(server_name);
}

}  // namespace scene_segmentation
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_SCENE_SEGMENTATION_TASK_H
