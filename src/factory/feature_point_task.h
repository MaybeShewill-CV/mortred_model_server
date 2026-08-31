#ifndef MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H
#define MORTRED_MODEL_SERVER_FEATURE_POINT_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/feature_point/superpoint.h"

namespace jinq {
namespace factory {
namespace feature_point {

using jinq::models::BaseAiModel;

using jinq::models::feature_point::SuperPoint;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_superpoint_extractor(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<SuperPoint<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::feature_point::std_feature_point_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable SuperPoint parameters. NOTE: nms_radius is a PIXEL
 * distance (not an IoU) - deliberately named differently from the detection
 * family's nms_threshold to keep cross-family key semantics honest */
inline const std::vector<jinq::models::backend::ParamSpec> &feature_point_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::f32("score_threshold").range(0.001, 1.0).desc("interest point confidence threshold"),
        jinq::models::backend::ParamSpec::i32("nms_radius").range(1, 50).desc("NMS suppression radius in pixels"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"SUPERPOINT", "Superpoint feature point detection", "SUPERPOINT_FP_SERVER", &create_superpoint_extractor<ImageInput, Output>,
              &jinq::server::response::fill_feature_points, feature_point_param_specs()},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_server(const std::string &model_section, const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), model_section, server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_superpoint_fp_server(const std::string &server_name) {
    return create_server("SUPERPOINT", server_name);
}

} // namespace feature_point
} // namespace factory
} // namespace jinq

#endif
