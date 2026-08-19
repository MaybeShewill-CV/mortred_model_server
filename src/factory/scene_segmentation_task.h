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
#include "server/generic_cv_server.h"
#include "server/response_serializers.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace scene_segmentation {

using jinq::models::scene_segmentation::BiseNetV2;
using jinq::models::scene_segmentation::HRNetSegmentation;
using jinq::models::scene_segmentation::MsOcrNet;
using jinq::models::scene_segmentation::PPHumanSeg;

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
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "BISENETV2_SERVER";
        spec.model_section = "BISENETV2";
        spec.display_name = "bisenetv2 segmentation";
        spec.make_worker = [](const std::string& name) {
            return create_bisenetv2_segmentor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_scene_segmentation;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create pphuman segmentation server
inline std::unique_ptr<BaseAiServer> create_pphuman_seg_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "PPHUMAN_SEG_SERVER";
        spec.model_section = "PPHUMAN_SEG";
        spec.display_name = "pphuman segmentation";
        spec.make_worker = [](const std::string& name) {
            return create_pphuman_segmentor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_scene_segmentation;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create hrnet scene segmentation server
inline std::unique_ptr<BaseAiServer> create_hrnet_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "HRNET_SERVER";
        spec.model_section = "HRNET";
        spec.display_name = "hrnet segmentation";
        spec.make_worker = [](const std::string& name) {
            return create_hrnet_segmentor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_scene_segmentation;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

}  // namespace scene_segmentation
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_SCENE_SEGMENTATION_TASK_H
