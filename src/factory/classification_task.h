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
#include "server/classification/densenet_server.h"
#include "server/classification/mobilenetv2_server.h"
#include "server/classification/resnet_server.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace classification {

using jinq::models::classification::DenseNet;
using jinq::models::classification::Dinov2;
using jinq::models::classification::MobileNetv2;
using jinq::models::classification::ResNet;
using jinq::server::classification::DenseNetServer;
using jinq::server::classification::MobileNetv2Server;
using jinq::server::classification::ResNetServer;

// create mobilenetv2 classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_mobilenetv2_classifier(const std::string& classifier_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<MobileNetv2<INPUT, OUTPUT> >(classifier_name);
    return model_factory.create(classifier_name);
}

// create mobilenetv2 classification server
inline std::unique_ptr<BaseAiServer> create_mobilenetv2_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<MobileNetv2Server>(server_name);
    return server_factory.create(server_name);
}

// create resnet classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_resnet_classifier(const std::string& classifier_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<ResNet<INPUT, OUTPUT> >(classifier_name);
    return model_factory.create(classifier_name);
}

// create resnet classification server
inline std::unique_ptr<BaseAiServer> create_resnet_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<ResNetServer>(server_name);
    return server_factory.create(server_name);
}

// create densenet classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_densenet_classifier(const std::string& classifier_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<DenseNet<INPUT, OUTPUT> >(classifier_name);
    return model_factory.create(classifier_name);
}

// create densenet classification server
inline std::unique_ptr<BaseAiServer> create_densenet_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_type<DenseNetServer>(server_name);
    return server_factory.create(server_name);
}

// create dinov2 classification model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_dinov2_classifier(const std::string& classifier_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<Dinov2<INPUT, OUTPUT> >(classifier_name);
    return model_factory.create(classifier_name);
}

}  // namespace classification
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H
