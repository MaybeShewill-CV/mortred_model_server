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

namespace mortred {
namespace factory {
namespace object_detection {

using mortred::models::BaseAiModel;

using mortred::models::object_detection::CenterFaceDetector;
using mortred::models::object_detection::LibFaceDetector;
using mortred::models::object_detection::NanoDetector;
using mortred::models::object_detection::YoloV5Detector;
using mortred::models::object_detection::YoloV6Detector;
using mortred::models::object_detection::YoloV7Detector;
using mortred::models::object_detection::YoloV8Detector;

// object detection and face detection have different output contracts, so they
// keep two typed catalogs instead of one type-erased list
using ObjectOutput = mortred::models::io_define::object_detection::std_object_detection_output;
using FaceOutput = mortred::models::io_define::object_detection::std_face_detection_output;
using mortred::server::ImageInput;
using ObjectEntry = mortred::factory::cv_catalog::CvModelEntry<ObjectOutput>;
using FaceEntry = mortred::factory::cv_catalog::CvModelEntry<FaceOutput>;

/*** request-overridable detection parameters (shared by every detector of
 * both catalogs); the TOML config stays the default source, a request can
 * only override within these ranges */
inline const std::vector<mortred::models::backend::ParamSpec> &detection_param_specs() {
    static const std::vector<mortred::models::backend::ParamSpec> specs = {
        mortred::models::backend::ParamSpec::f32("score_threshold").range(0.0, 1.0).desc("confidence threshold"),
        mortred::models::backend::ParamSpec::f32("nms_threshold").range(0.1, 1.0).desc("per-class NMS IoU threshold"),
        mortred::models::backend::ParamSpec::i32("top_k").range(1, 10000).desc("keep at most k detections"),
    };
    return specs;
}

inline const std::vector<ObjectEntry> &catalog() {
    static const std::vector<ObjectEntry> entries = {
        ObjectEntry{"yolov5l", "YOLOv5l object detection", "YOLOV5_DETECTION_SERVER", [] { return std::make_unique<YoloV5Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov5"},
        ObjectEntry{"yolov6s", "YOLOv6s object detection", "YOLOV6_DETECTION_SERVER", [] { return std::make_unique<YoloV6Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov6"},
        ObjectEntry{"nanodet_1x5", "NanoDet 1.5x object detection", "NANODET_DETECTION_SERVER",
                    [] { return std::make_unique<NanoDetector<ImageInput, ObjectOutput>>(); }, &mortred::server::response::fill_object_detection,
                    detection_param_specs(), "nanodet"},
        ObjectEntry{"nanodet_416", "NanoDet 416 object detection", "NANODET_DETECTION_SERVER",
                    [] { return std::make_unique<NanoDetector<ImageInput, ObjectOutput>>(); }, &mortred::server::response::fill_object_detection,
                    detection_param_specs(), "nanodet"},
        ObjectEntry{"yolov7", "YOLOv7 object detection", "YOLOV7_DETECTION_SERVER", [] { return std::make_unique<YoloV7Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov7"},
        ObjectEntry{"yolov7x", "YOLOv7x object detection", "YOLOV7_DETECTION_SERVER", [] { return std::make_unique<YoloV7Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov7"},
        ObjectEntry{"yolov8n", "YOLOv8n object detection", "YOLOV8_DETECTION_SERVER", [] { return std::make_unique<YoloV8Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov8"},
        ObjectEntry{"yolov8s", "YOLOv8s object detection", "YOLOV8_DETECTION_SERVER", [] { return std::make_unique<YoloV8Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov8"},
        ObjectEntry{"yolov8l", "YOLOv8l object detection", "YOLOV8_DETECTION_SERVER", [] { return std::make_unique<YoloV8Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov8"},
        ObjectEntry{"yolov8x", "YOLOv8x object detection", "YOLOV8_DETECTION_SERVER", [] { return std::make_unique<YoloV8Detector<ImageInput, ObjectOutput>>(); },
                    &mortred::server::response::fill_object_detection, detection_param_specs(), "yolov8"},
    };
    return entries;
}

inline const std::vector<FaceEntry> &face_catalog() {
    static const std::vector<FaceEntry> entries = {
        FaceEntry{"libface", "YuNet face detection", "LIBFACE_DETECTION_SERVER", [] { return std::make_unique<LibFaceDetector<ImageInput, FaceOutput>>(); },
                  &mortred::server::response::fill_face_detection, detection_param_specs(), "libface"},
        FaceEntry{"centerface", "CenterFace detection", "CENTER_FACE_DETECTION_SERVER",
                  [] { return std::make_unique<CenterFaceDetector<ImageInput, FaceOutput>>(); }, &mortred::server::response::fill_face_detection,
                  detection_param_specs(), "centerface"},
    };
    return entries;
}

} // namespace object_detection
} // namespace factory
} // namespace mortred

#endif
