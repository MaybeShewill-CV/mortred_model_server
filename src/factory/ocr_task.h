#ifndef MORTRED_MODEL_SERVER_OCR_TASK_H
#define MORTRED_MODEL_SERVER_OCR_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/ocr/db_text_detector.h"

namespace jinq {
namespace factory {
namespace ocr {

using jinq::models::BaseAiModel;

using jinq::models::ocr::DBTextDetector;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_dbtext_detector() {
    return std::make_unique<DBTextDetector<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::ocr::std_text_regions_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable DBNet text detection parameters */
inline const std::vector<jinq::models::backend::ParamSpec> &ocr_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::f32("score_threshold").range(0.1, 0.9).desc("text region confidence threshold"),
        jinq::models::backend::ParamSpec::i32("top_k").range(1, 10000).desc("keep at most k text regions"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"DBNET", "dbnet", "DBNET_SERVER", &create_dbtext_detector<ImageInput, Output>, &jinq::server::response::fill_text_regions,
              ocr_param_specs()},
    };
    return entries;
}

} // namespace ocr
} // namespace factory
} // namespace jinq

#endif
