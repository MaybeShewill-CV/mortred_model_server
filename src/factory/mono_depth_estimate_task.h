#ifndef MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H
#define MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/mono_depth_estimation/depth_anything.h"
#include "models/mono_depth_estimation/metric3d.h"

namespace mortred {
namespace factory {
namespace mono_depth_estimation {

using mortred::models::BaseAiModel;

using mortred::models::mono_depth_estimation::DepthAnything;
using mortred::models::mono_depth_estimation::Metric3D;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_metric3d_estimator() {
    return std::make_unique<Metric3D<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_depth_anything_estimator() {
    return std::make_unique<DepthAnything<INPUT, OUTPUT>>();
}

using Output = mortred::models::io_define::mono_depth_estimation::std_mde_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"depth_vits14", "Depth Anything ViT-S/14", "DEPTH_ANYTHING_ESTIMATION_SERVER",
              &create_depth_anything_estimator<ImageInput, Output>, &mortred::server::response::fill_depth_estimation, {},
              "depth_anything"},
        Entry{"depth_vitb14", "Depth Anything ViT-B/14", "DEPTH_ANYTHING_ESTIMATION_SERVER",
              &create_depth_anything_estimator<ImageInput, Output>, &mortred::server::response::fill_depth_estimation, {},
              "depth_anything"},
        Entry{"depth_vitl14", "Depth Anything ViT-L/14", "DEPTH_ANYTHING_ESTIMATION_SERVER",
              &create_depth_anything_estimator<ImageInput, Output>, &mortred::server::response::fill_depth_estimation, {},
              "depth_anything"},
        Entry{"metric3d_512", "Metric3D 512x1088", "METRIC3D_ESTIMATION_SERVER", &create_metric3d_estimator<ImageInput, Output>,
              &mortred::server::response::fill_depth_estimation, {}, "metric3d"},
        Entry{"metric3d_1088", "Metric3D 1088x1920", "METRIC3D_ESTIMATION_SERVER", &create_metric3d_estimator<ImageInput, Output>,
              &mortred::server::response::fill_depth_estimation, {}, "metric3d"},
    };
    return entries;
}

} // namespace mono_depth_estimation
} // namespace factory
} // namespace mortred

#endif
