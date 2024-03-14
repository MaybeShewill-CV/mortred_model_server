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
#include "models/object_detection/libface_detector.h"
#include "models/object_detection/nano_detector.h"
#include "models/object_detection/yolov5_detector.h"
#include "models/object_detection/yolov6_detector.h"
#include "models/object_detection/yolov7_detector.h"
#include "models/object_detection/yolov8_detector.h"
#include "models/object_detection/centerface_detector.h"
#include "server/object_detection/libface_det_server.h"
#include "server/object_detection/nano_det_server.h"
#include "server/object_detection/yolov5_det_server.h"
#include "server/object_detection/yolov6_det_server.h"
#include "server/object_detection/yolov7_det_server.h"
#include "server/object_detection/yolov8_det_server.h"
#include "server/object_detection/centerface_det_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace object_detection {
using jinq::models::object_detection::LibFaceDetector;
using jinq::models::object_detection::NanoDetector;
using jinq::models::object_detection::YoloV5Detector;
using jinq::models::object_detection::YoloV6Detector;
using jinq::models::object_detection::YoloV7Detector;
using jinq::models::object_detection::YoloV8Detector;
using jinq::models::object_detection::CenterFaceDetector;

using jinq::server::object_detection::LibfaceDetServer;
using jinq::server::object_detection::NanoDetServer;
using jinq::server::object_detection::YoloV5DetServer;
using jinq::server::object_detection::YoloV6DetServer;
using jinq::server::object_detection::YoloV7DetServer;
using jinq::server::object_detection::YoloV8DetServer;
using jinq::server::object_detection::CenterfaceDetServer;

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
 * create yolov6 object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov6_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(YoloV6Detector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(detector_name);
}

/***
 * create yolov6 object detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_yolov6_det_server(const std::string& server_name) {
    REGISTER_AI_SERVER(YoloV6DetServer, server_name)
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

/***
 * create yolov7 object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov7_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(YoloV7Detector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(detector_name);
}

/***
 * create yolov7 object detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_yolov7_det_server(const std::string& server_name) {
    REGISTER_AI_SERVER(YoloV7DetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create yolov8 object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov8_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(YoloV8Detector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(detector_name);
}

/***
 * create yolov8 object detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_yolov8_det_server(const std::string& server_name) {
    REGISTER_AI_SERVER(YoloV8DetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

/***
 * create center face object detection instance
 * @tparam INPUT
 * @tparam OUTPUT
 * @param detector_name
 * @return
 */
template <typename INPUT, typename OUTPUT>
static std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_centerface_detector(const std::string& detector_name) {
    REGISTER_AI_MODEL(CenterFaceDetector, detector_name, INPUT, OUTPUT)
    return ModelFactory<BaseAiModel<INPUT, OUTPUT>>::get_instance().get_model(detector_name);
}

/***
 * create center face object detection server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_centerface_det_server(const std::string& server_name) {
    REGISTER_AI_SERVER(CenterfaceDetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

} // namespace object_detection
} // namespace factory
} // namespace jinq

#endif // MM_AI_SERVER_OBJ_DETECTION_TASK_H
