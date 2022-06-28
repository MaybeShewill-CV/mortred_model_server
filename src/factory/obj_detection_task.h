/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: image_object_detection_task.h
 * Date: 22-6-8
 ************************************************/

#ifndef MM_AI_SERVER_OBJ_DETECTION_TASK_H
#define MM_AI_SERVER_OBJ_DETECTION_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "models/object_detection/nano_detector.h"
#include "models/object_detection/yolov5_detector.h"
#include "models/object_detection/libface_detector.h"
#include "server/object_detection/nano_det_server.h"
#include "server/object_detection/yolov5_det_server.h"
#include "server/object_detection/libface_det_server.h"

namespace morted {
namespace factory {

using morted::factory::ModelFactory;
using morted::models::BaseAiModel;
using morted::server::BaseAiServer;

namespace object_detection {
using morted::models::object_detection::NanoDetector;
using morted::models::object_detection::YoloV5Detector;
using morted::models::object_detection::LibFaceDetector;

using morted::server::object_detection::NanoDetServer;
using morted::server::object_detection::YoloV5DetServer;
using morted::server::object_detection::LibfaceDetServer;

/***
 * create yolov5 object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov5_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(YoloV5Detector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(detector_name);
}

/***
 * create yolov5 object detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_yolov5_det_server(const std::string& server_name) {
    REGISTER_AI_SERVER(YoloV5DetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create nanodet object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_nanodet_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(NanoDetector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(detector_name);
}

/***
 * create nanodet object detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_nanodet_det_server(const std::string& server_name) {
    REGISTER_AI_SERVER(NanoDetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create libface object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_libface_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(LibFaceDetector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(detector_name);
}

/***
 * create libface detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_libface_det_server(const std::string& server_name) {
    REGISTER_AI_SERVER(LibfaceDetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

} // namespace object_detection
} // namespace factory
} // namespace morted

#endif // MM_AI_SERVER_OBJ_DETECTION_TASK_H
