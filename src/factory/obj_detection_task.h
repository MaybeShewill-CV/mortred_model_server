/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: image_object_detection_task.h
* Date: 22-6-8
************************************************/

#ifndef MORTRED_MODEL_SERVER_OBJ_DETECTION_TASK_H
#define MORTRED_MODEL_SERVER_OBJ_DETECTION_TASK_H

#include <memory>
#include <string>

#include "factory/base_factory.h"
#include "models/base_model.h"
#include "models/object_detection/centerface_detector.h"
#include "models/object_detection/libface_detector.h"
#include "models/object_detection/nano_detector.h"
#include "models/object_detection/yolov5_detector.h"
#include "models/object_detection/yolov6_detector.h"
#include "models/object_detection/yolov7_detector.h"
#include "models/object_detection/yolov8_detector.h"
#include "server/abstract_server.h"
#include "server/generic_ai_server.h"
#include "server/response_serializers.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace object_detection {

using jinq::models::object_detection::CenterFaceDetector;
using jinq::models::object_detection::LibFaceDetector;
using jinq::models::object_detection::NanoDetector;
using jinq::models::object_detection::YoloV5Detector;
using jinq::models::object_detection::YoloV6Detector;
using jinq::models::object_detection::YoloV7Detector;
using jinq::models::object_detection::YoloV8Detector;

// create yolov5 object detection model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_yolov5_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<YoloV5Detector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create yolov5 object detection server
inline std::unique_ptr<BaseAiServer> create_yolov5_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::object_detection::std_object_detection_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "YOLOV5_DETECTION_SERVER";
        spec.model_section = "YOLOV5";
        spec.display_name = "Yolov5 object detection";
        spec.make_worker = [](const std::string& name) {
            return create_yolov5_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_object_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create yolov6 object detection model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_yolov6_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<YoloV6Detector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create yolov6 object detection server
inline std::unique_ptr<BaseAiServer> create_yolov6_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::object_detection::std_object_detection_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "YOLOV6_DETECTION_SERVER";
        spec.model_section = "YOLOV6";
        spec.display_name = "Yolov6 object detection";
        spec.make_worker = [](const std::string& name) {
            return create_yolov6_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_object_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create nanodet object detection model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_nanodet_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<NanoDetector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create nanodet object detection server
inline std::unique_ptr<BaseAiServer> create_nanodet_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::object_detection::std_object_detection_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "NANODET_DETECTION_SERVER";
        spec.model_section = "NANODET";
        spec.display_name = "NanoDet object detection";
        spec.make_worker = [](const std::string& name) {
            return create_nanodet_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_object_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create libface detection model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_libface_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<LibFaceDetector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create libface detection server
inline std::unique_ptr<BaseAiServer> create_libface_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::object_detection::std_face_detection_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "LIBFACE_DETECTION_SERVER";
        spec.model_section = "LIBFACE";
        spec.display_name = "libface object detection";
        spec.make_worker = [](const std::string& name) {
            return create_libface_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_face_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create yolov7 object detection model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_yolov7_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<YoloV7Detector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create yolov7 object detection server
inline std::unique_ptr<BaseAiServer> create_yolov7_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::object_detection::std_object_detection_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "YOLOV7_DETECTION_SERVER";
        spec.model_section = "YOLOV7";
        spec.display_name = "Yolov7 object detection";
        spec.make_worker = [](const std::string& name) {
            return create_yolov7_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_object_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create yolov8 object detection model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_yolov8_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<YoloV8Detector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create yolov8 object detection server
inline std::unique_ptr<BaseAiServer> create_yolov8_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::object_detection::std_object_detection_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "YOLOV8_DETECTION_SERVER";
        spec.model_section = "YOLOV8";
        spec.display_name = "Yolov8 object detection";
        spec.make_worker = [](const std::string& name) {
            return create_yolov8_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_object_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

// create centerface detection model
template<typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT> > create_centerface_detector(const std::string& detector_name) {
    auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
    model_factory.template register_type<CenterFaceDetector<INPUT, OUTPUT> >(detector_name);
    return model_factory.create(detector_name);
}

// create centerface detection server
inline std::unique_ptr<BaseAiServer> create_centerface_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::object_detection::std_face_detection_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "CENTER_FACE_DETECTION_SERVER";
        spec.model_section = "CENTER_FACE";
        spec.display_name = "center face object detection";
        spec.make_worker = [](const std::string& name) {
            return create_centerface_detector<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_face_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}

}  // namespace object_detection
}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_OBJ_DETECTION_TASK_H
