/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: image_ocr_task.h
* Date: 22-6-8
************************************************/

#ifndef MORTRED_MODEL_SERVER_OCR_TASK_H
#define MORTRED_MODEL_SERVER_OCR_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/ocr/db_text_detector.h"
#include "server/ocr/dbnet_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace ocr {
using jinq::models::ocr::DBTextDetector;

using jinq::server::ocr::DBNetServer;

/***
 * create db text detector instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template<typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_dbtext_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(DBTextDetector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance().get_model(detector_name);
}

/***
 * create dbnet text region detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_dbtext_detection_server(const std::string& server_name) {
    REGISTER_AI_SERVER(DBNetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

}
}
}

#endif //MORTRED_MODEL_SERVER_OCR_TASK_H
