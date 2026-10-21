#ifndef MORTRED_MODEL_SERVER_OCR_TASK_H
#define MORTRED_MODEL_SERVER_OCR_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/ocr/db_text_detector.h"

namespace mortred {
namespace factory {
namespace ocr {

using mortred::models::BaseAiModel;

using mortred::models::ocr::DBTextDetector;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_dbtext_detector() {
    return std::make_unique<DBTextDetector<INPUT, OUTPUT>>();
}

using Output = mortred::models::io_define::ocr::std_text_regions_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable DBNet text detection parameters */
inline const std::vector<mortred::models::backend::ParamSpec> &ocr_param_specs() {
    static const std::vector<mortred::models::backend::ParamSpec> specs = {
        mortred::models::backend::ParamSpec::f32("score_threshold").range(0.1, 0.9).desc("text region confidence threshold"),
        mortred::models::backend::ParamSpec::i32("top_k").range(1, 10000).desc("keep at most k text regions"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"dbnet", "DBNet text detection", "DBNET_SERVER", &create_dbtext_detector<ImageInput, Output>,
              &mortred::server::response::fill_text_regions, ocr_param_specs(), "dbnet"},
    };
    return entries;
}

} // namespace ocr
} // namespace factory
} // namespace mortred

#endif
