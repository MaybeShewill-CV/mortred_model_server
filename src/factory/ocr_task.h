/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: image_ocr_task.h
* Date: 22-6-8
************************************************/

#ifndef MORTRED_MODEL_SERVER_OCR_TASK_H
#define MORTRED_MODEL_SERVER_OCR_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/ocr/db_text_detector.h"
#include "server/abstract_server.h"
#include "server/generic_ai_server.h"
#include "server/response_serializers.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace ocr {

using jinq::models::ocr::DBTextDetector;

// create db text detector model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_dbtext_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<DBTextDetector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create dbnet text region detection server
inline std::unique_ptr<BaseAiServer> create_dbtext_detection_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::ocr::std_text_regions_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "DBNET_SERVER";
        spec.model_section = "DBNET";
        spec.display_name = "dbnet";
        spec.make_worker = [](const std::string& name) {
            return create_dbtext_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_text_regions;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

}  // namespace ocr
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_OCR_TASK_H
