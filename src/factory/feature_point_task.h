#ifndef MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H
#define MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "factory/model_catalog.h"
#include "models/feature_point/lightglue.h"
#include "models/feature_point/superpoint.h"
#include "models/model_io_define.h"

namespace mortred {
namespace factory {
namespace feature_point {

using mortred::models::BaseAiModel;

using mortred::models::feature_point::LightGlue;
using mortred::models::feature_point::SuperPoint;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_superpoint_extractor() {
    return std::make_unique<SuperPoint<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_lightglue_matcher() {
    return std::make_unique<LightGlue<INPUT, OUTPUT>>();
}

using Output = mortred::models::io_define::feature_point::std_feature_point_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable SuperPoint parameters. NOTE: nms_radius is a PIXEL
 * distance (not an IoU) - deliberately named differently from the detection
 * family's nms_threshold to keep cross-family key semantics honest */
inline const std::vector<mortred::models::backend::ParamSpec> &feature_point_param_specs() {
    static const std::vector<mortred::models::backend::ParamSpec> specs = {
        mortred::models::backend::ParamSpec::f32("score_threshold").range(0.001, 1.0).desc("interest point confidence threshold"),
        mortred::models::backend::ParamSpec::i32("nms_radius").range(1, 50).desc("NMS suppression radius in pixels"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"superpoint", "SuperPoint feature detection", "SUPERPOINT_FP_SERVER",
              &create_superpoint_extractor<ImageInput, Output>, &mortred::server::response::fill_feature_points,
              feature_point_param_specs(), "superpoint"},
    };
    return entries;
}

using MatchInput = mortred::models::io_define::common_io::pair_mat_input;
using MatchOutput = mortred::models::io_define::feature_point::std_feature_point_match_output;
using MatchEntry = mortred::factory::model_catalog::ModelCatalogEntry<MatchInput, MatchOutput>;

inline const std::vector<MatchEntry> &match_catalog() {
    static const std::vector<MatchEntry> entries = {
        MatchEntry{"LIGHTGLUE", "LightGlue feature matcher", &create_lightglue_matcher<MatchInput, MatchOutput>},
    };
    return entries;
}

} // namespace feature_point
} // namespace factory
} // namespace mortred

#endif
