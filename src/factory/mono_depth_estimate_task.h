#ifndef MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H
#define MORTRED_MODEL_SERVER_MONO_DEPTH_ESTIMATE_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/mono_depth_estimation/depth_anything.h"
#include "models/mono_depth_estimation/metric3d.h"

namespace jinq {
namespace factory {
namespace mono_depth_estimation {

using jinq::models::BaseAiModel;

using jinq::models::mono_depth_estimation::DepthAnything;
using jinq::models::mono_depth_estimation::Metric3D;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_metric3d_estimator(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<Metric3D<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_depth_anything_estimator(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<DepthAnything<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::mono_depth_estimation::std_mde_output;
using jinq::server::Base64Input;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"METRIC3D", "metric3d estimation", "METRIC3D_ESTIMATION_SERVER", &create_metric3d_estimator<Base64Input, Output>,
              &jinq::server::response::fill_depth_estimation},
        Entry{"DEPTH_ANYTHING", "depth anything estimation", "DEPTH_ANYTHING_ESTIMATION_SERVER",
              &create_depth_anything_estimator<Base64Input, Output>, &jinq::server::response::fill_depth_estimation},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_server(const std::string &model_section, const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), model_section, server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_metric3d_estimation_server(const std::string &server_name) {
    return create_server("METRIC3D", server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_depth_anything_estimation_server(const std::string &server_name) {
    return create_server("DEPTH_ANYTHING", server_name);
}

} // namespace mono_depth_estimation
} // namespace factory
} // namespace jinq

#endif
