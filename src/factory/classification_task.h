/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: classification_task.h
* Date: 22-6-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H
#define MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/classification/densenet.h"
#include "models/classification/dinov2.h"
#include "models/classification/mobilenetv2.h"
#include "models/classification/resnet.h"
#include "server/abstract_server.h"
#include "server/generic_cv_server.h"
#include "server/response_serializers.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace classification {

using jinq::models::classification::DenseNet;
using jinq::models::classification::Dinov2;
using jinq::models::classification::MobileNetv2;
using jinq::models::classification::ResNet;

// create mobilenetv2 classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_mobilenetv2_classifier(const std::string& classifier_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)classifier_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new MobileNetv2<INPUT, OUTPUT>());
}

// create mobilenetv2 classification server
inline std::unique_ptr<BaseAiServer> create_mobilenetv2_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::classification::std_classification_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "MOBILENETV2_CLASSIFICATION_SERVER";
        spec.model_section = "MOBILENETV2";
        spec.display_name = "Mobilenetv2 classification";
        spec.make_worker = [](const std::string& name) {
            return create_mobilenetv2_classifier<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_classification;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create resnet classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_resnet_classifier(const std::string& classifier_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)classifier_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new ResNet<INPUT, OUTPUT>());
}

// create resnet classification server
inline std::unique_ptr<BaseAiServer> create_resnet_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::classification::std_classification_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "RESNET_CLASSIFICATION_SERVER";
        spec.model_section = "RESNET";
        spec.display_name = "Resnet classification";
        spec.make_worker = [](const std::string& name) {
            return create_resnet_classifier<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_classification;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create densenet classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_densenet_classifier(const std::string& classifier_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)classifier_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new DenseNet<INPUT, OUTPUT>());
}

// create densenet classification server
inline std::unique_ptr<BaseAiServer> create_densenet_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::classification::std_classification_output;
        jinq::server::CvServerSpec<Output> spec;
        spec.server_section = "DENSENET_CLASSIFICATION_SERVER";
        spec.model_section = "DENSENET";
        spec.display_name = "densenet classification";
        spec.make_worker = [](const std::string& name) {
            return create_densenet_classifier<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_classification;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::CvModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create dinov2 classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_dinov2_classifier(const std::string& classifier_name) {
    // Direct construction: no global registry writes (no side effects or mutex
    // overhead); avoids re-registering on every create. name kept for compatibility.
    (void)classifier_name;
    return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new Dinov2<INPUT, OUTPUT>());
}

}  // namespace classification
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H
