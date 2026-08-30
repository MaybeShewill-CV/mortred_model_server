#ifndef MORTRED_MODEL_SERVER_OBJ_DETECTION_TASK_H
#define MORTRED_MODEL_SERVER_OBJ_DETECTION_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/object_detection/centerface_detector.h"
#include "models/object_detection/libface_detector.h"
#include "models/object_detection/nano_detector.h"
#include "models/object_detection/yolov5_detector.h"
#include "models/object_detection/yolov6_detector.h"
#include "models/object_detection/yolov7_detector.h"
#include "models/object_detection/yolov8_detector.h"

namespace jinq {
namespace factory {
namespace object_detection {

using jinq::models::BaseAiModel;

using jinq::models::object_detection::CenterFaceDetector;
using jinq::models::object_detection::LibFaceDetector;
using jinq::models::object_detection::NanoDetector;
using jinq::models::object_detection::YoloV5Detector;
using jinq::models::object_detection::YoloV6Detector;
using jinq::models::object_detection::YoloV7Detector;
using jinq::models::object_detection::YoloV8Detector;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov5_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<YoloV5Detector<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov6_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<YoloV6Detector<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_nanodet_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<NanoDetector<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_libface_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<LibFaceDetector<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov7_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<YoloV7Detector<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_yolov8_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<YoloV8Detector<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_centerface_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<CenterFaceDetector<INPUT, OUTPUT>>();
}

// object detection and face detection have different output contracts, so they
// keep two typed catalogs instead of one type-erased list
using ObjectOutput = jinq::models::io_define::object_detection::std_object_detection_output;
using FaceOutput = jinq::models::io_define::object_detection::std_face_detection_output;
using jinq::server::Base64Input;
using ObjectEntry = jinq::factory::cv_catalog::CvModelEntry<ObjectOutput>;
using FaceEntry = jinq::factory::cv_catalog::CvModelEntry<FaceOutput>;

inline const std::vector<ObjectEntry> &catalog() {
    static const std::vector<ObjectEntry> entries = {
        ObjectEntry{"YOLOV5", "Yolov5 object detection", "YOLOV5_DETECTION_SERVER", &create_yolov5_detector<Base64Input, ObjectOutput>,
                    &jinq::server::response::fill_object_detection},
        ObjectEntry{"YOLOV6", "Yolov6 object detection", "YOLOV6_DETECTION_SERVER", &create_yolov6_detector<Base64Input, ObjectOutput>,
                    &jinq::server::response::fill_object_detection},
        ObjectEntry{"NANODET", "nanodet object detection", "NANODET_DETECTION_SERVER", &create_nanodet_detector<Base64Input, ObjectOutput>,
                    &jinq::server::response::fill_object_detection},
        ObjectEntry{"YOLOV7", "Yolov7 object detection", "YOLOV7_DETECTION_SERVER", &create_yolov7_detector<Base64Input, ObjectOutput>,
                    &jinq::server::response::fill_object_detection},
        ObjectEntry{"YOLOV8", "Yolov8 object detection", "YOLOV8_DETECTION_SERVER", &create_yolov8_detector<Base64Input, ObjectOutput>,
                    &jinq::server::response::fill_object_detection},
    };
    return entries;
}

inline const std::vector<FaceEntry> &face_catalog() {
    static const std::vector<FaceEntry> entries = {
        FaceEntry{"LIBFACE", "libface face detection", "LIBFACE_DETECTION_SERVER", &create_libface_detector<Base64Input, FaceOutput>,
                  &jinq::server::response::fill_face_detection},
        FaceEntry{"CENTER_FACE", "center face detection", "CENTER_FACE_DETECTION_SERVER",
                  &create_centerface_detector<Base64Input, FaceOutput>, &jinq::server::response::fill_face_detection},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_server(const std::string &model_section, const std::string &server_name) {
    if (jinq::factory::cv_catalog::find_entry(catalog(), model_section) != nullptr) {
        return jinq::factory::cv_catalog::create_server(catalog(), model_section, server_name);
    }
    return jinq::factory::cv_catalog::create_server(face_catalog(), model_section, server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_yolov5_det_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), "YOLOV5", server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_yolov6_det_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), "YOLOV6", server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_nanodet_det_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), "NANODET", server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_yolov7_det_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), "YOLOV7", server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_yolov8_det_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), "YOLOV8", server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_libface_det_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(face_catalog(), "LIBFACE", server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_centerface_det_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(face_catalog(), "CENTER_FACE", server_name);
}

} // namespace object_detection
} // namespace factory
} // namespace jinq

#endif
