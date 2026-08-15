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
#include "server/enhancement/attentive_gan_derain_server.h"
#include "server/enhancement/enlighten_gan_server.h"
#include "server/enhancement/real_esr_gan_server.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace enhancement {

using jinq::models::enhancement::AttentiveGanDerain;
using jinq::models::enhancement::EnlightenGan;
using jinq::models::enhancement::RealEsrGan;
using jinq::server::enhancement::AttentiveGanDerainServer;
using jinq::server::enhancement::EnlightenGanServer;
using jinq::server::enhancement::RealEsrGanServer;

// create enlighten-gan low light enhancement model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_enlightengan_enhancementor(const std::string& enhancementor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<EnlightenGan<INPUT, OUTPUT> >(enhancementor_name);
    return model_factory.create(enhancementor_name);
}

// create enlighten gan enhancement server
inline std::unique_ptr<BaseAiServer> create_enlightengan_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<EnlightenGanServer>(server_name);
    return server_factory.create(server_name);
}

// create attentive gan derain model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_attentivegan_enhancementor(const std::string& enhancementor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<AttentiveGanDerain<INPUT, OUTPUT> >(enhancementor_name);
    return model_factory.create(enhancementor_name);
}

// create attentive gan derain server
inline std::unique_ptr<BaseAiServer> create_attentivegan_derain_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<AttentiveGanDerainServer>(server_name);
    return server_factory.create(server_name);
}

// create real esrgan super resolution model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_realesrgan_enhancementor(const std::string& enhancementor_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<RealEsrGan<INPUT, OUTPUT> >(enhancementor_name);
    return model_factory.create(enhancementor_name);
}

// create real esrgan super resolution server
inline std::unique_ptr<BaseAiServer> create_realesrgan_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<RealEsrGanServer>(server_name);
    return server_factory.create(server_name);
}

}  // namespace enhancement
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H
