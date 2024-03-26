/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: classification_task.h
* Date: 22-6-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H
#define MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/classification/mobilenetv2.h"
#include "models/classification/resnet.h"
#include "models/classification/densenet.h"
#include "models/classification/dinov2.h"
#include "server/classification/mobilenetv2_server.h"
#include "server/classification/resnet_server.h"
#include "server/classification/densenet_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::factory::ServerFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace classification {
using jinq::models::classification::MobileNetv2;
using jinq::models::classification::ResNet;
using jinq::models::classification::DenseNet;
using jinq::models::classification::Dinov2;

using jinq::server::classification::MobileNetv2Server;
using jinq::server::classification::ResNetServer;
using jinq::server::classification::DenseNetServer;

/***
 * create mobilenetv2 image classification
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_mobilenetv2_classifier(
    const std::string& classifier_name) {
    REGISTER_AI_MODEL(MobileNetv2, classifier_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(classifier_name);
}

/***
 * create mobilenetv2 image classification server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_mobilenetv2_cls_server(const std::string& server_name) {
    REGISTER_AI_SERVER(MobileNetv2Server, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create resnet image classification
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_resnet_classifier(
    const std::string& classifier_name) {
    REGISTER_AI_MODEL(ResNet, classifier_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(classifier_name);
}

/***
 * create resnet image classification server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_resnet_cls_server(const std::string& server_name) {
    REGISTER_AI_SERVER(ResNetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create densenet image classification
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_densenet_classifier(
    const std::string& classifier_name) {
    REGISTER_AI_MODEL(DenseNet, classifier_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(classifier_name);
}

/***
 * create densenet image classification server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_densenet_cls_server(const std::string& server_name) {
    REGISTER_AI_SERVER(DenseNetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create dinov2 image classification
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_dinov2_classifier(
    const std::string& classifier_name) {
    REGISTER_AI_MODEL(Dinov2, classifier_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(classifier_name);
}

}
}
}

#endif //MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H
