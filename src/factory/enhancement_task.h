/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: enhancement_task.h
* Date: 22-6-13
************************************************/

#ifndef MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H
#define MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/enhancement/attentive_gan_derain_net.h"
#include "models/enhancement/enlightengan.h"
#include "models/enhancement/real_esrgan.h"
#include "server/abstract_server.h"
#include "server/generic_cv_server.h"
#include "server/response_serializers.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace enhancement {

using jinq::models::enhancement::AttentiveGanDerain;
using jinq::models::enhancement::EnlightenGan;
using jinq::models::enhancement::RealEsrGan;

// create enlighten-gan low light enhancement model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_enlightengan_enhancementor(const std::string& enhancementor_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)enhancementor_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new EnlightenGan<INPUT, OUTPUT>());
}

// create enlighten gan enhancement server
inline std::unique_ptr<BaseAiServer> create_enlightengan_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::enhancement::std_enhancement_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "ENLIGHTEN_GAN_SERVER";
        spec.model_section = "ENLIGHTEN_GAN";
        spec.display_name = "enlighten gan";
        spec.make_worker = [](const std::string& name) {
            return create_enlightengan_enhancementor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_enhancement;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create attentive gan derain model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_attentivegan_enhancementor(const std::string& enhancementor_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)enhancementor_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new AttentiveGanDerain<INPUT, OUTPUT>());
}

// create attentive gan derain server
inline std::unique_ptr<BaseAiServer> create_attentivegan_derain_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::enhancement::std_enhancement_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "ATTENTIVE_GAN_DERAIN_SERVER";
        spec.model_section = "ATTENTIVE_GAN_DERAIN";
        spec.display_name = "attentive gan derain";
        spec.make_worker = [](const std::string& name) {
            return create_attentivegan_enhancementor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_enhancement;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create real esrgan super resolution model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_realesrgan_enhancementor(const std::string& enhancementor_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)enhancementor_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new RealEsrGan<INPUT, OUTPUT>());
}

// create real esrgan super resolution server
inline std::unique_ptr<BaseAiServer> create_realesrgan_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::enhancement::std_enhancement_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "REAL_ESRGAN_SERVER";
        spec.model_section = "REAL_ESRGAN";
        spec.display_name = "real esr-gan";
        spec.make_worker = [](const std::string& name) {
            return create_realesrgan_enhancementor<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_enhancement;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

}  // namespace enhancement
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H
