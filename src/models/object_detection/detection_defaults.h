#ifndef MORTRED_MODELS_OBJECT_DETECTION_DETECTION_DEFAULTS_H
#define MORTRED_MODELS_OBJECT_DETECTION_DETECTION_DEFAULTS_H

#include <cstdint>

namespace mortred {
namespace models {
namespace object_detection {

/*** Single source of the fallback detection thresholds each detector's
 * on_init() applies when the TOML config omits them (all request-overridable
 * through the ParamSpecs in factory/obj_detection_task.h). These lived as
 * copy-pasted literals in seven detector .inl files before this header. */
struct DetectionDefaults {
    float score_threshold;
    float nms_threshold;
    int32_t keep_top_k;
    int32_t class_nums;  // 0 = the detector carries no class count
};

// COCO-trained object detectors (yolov5/6/7/8, nanodet)
inline constexpr DetectionDefaults coco_object_defaults{0.4f, 0.35f, 250, 80};
// face detectors (libface/YuNet, centerface)
inline constexpr DetectionDefaults face_defaults{0.6f, 0.3f, 250, 0};

}  // namespace object_detection
}  // namespace models
}  // namespace mortred

#endif  // MORTRED_MODELS_OBJECT_DETECTION_DETECTION_DEFAULTS_H
