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
#include "server/ocr/dbnet_server.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace ocr {

using jinq::models::ocr::DBTextDetector;
using jinq::server::ocr::DBNetServer;

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
    server_factory.register_type<DBNetServer>(server_name);
    return server_factory.create(server_name);
}

}  // namespace ocr
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_OCR_TASK_H
