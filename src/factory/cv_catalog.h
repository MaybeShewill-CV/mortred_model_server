#ifndef MORTRED_FACTORY_CV_CATALOG_H
#define MORTRED_FACTORY_CV_CATALOG_H

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "factory/base_factory.h"
#include "glog/logging.h"
#include "models/backend/param_spec.h"
#include "models/base_model.h"
#include "models/catalog/model_entry.h"
#include "server/generic_cv_server.h"
#include "server/response_serializers.h"

namespace mortred {
namespace factory {
namespace cv_catalog {

template <typename OUTPUT> struct CvModelEntry : public mortred::models::catalog::ServedModelEntry {
    mortred::server::CvWorkerFactory<OUTPUT> make_worker;
    mortred::server::CvResponseFiller<OUTPUT> fill_response = nullptr;
    std::vector<mortred::models::backend::ParamSpec> param_specs;
    // Weight family (yolov8, dinov2, …). Catalog id is model_section and is
    // lowercase. Empty means this row is not part of the static_bs1 split.
    std::string family;
};

template <typename OUTPUT>
const CvModelEntry<OUTPUT> *find_entry(const std::vector<CvModelEntry<OUTPUT>> &entries, const std::string &model_section) {
    const auto found = std::find_if(entries.begin(), entries.end(),
                                    [&model_section](const CvModelEntry<OUTPUT> &entry) { return entry.model_section == model_section; });
    return found == entries.end() ? nullptr : &*found;
}

template <typename OUTPUT>
std::unique_ptr<mortred::server::BaseAiServer> create_server(const CvModelEntry<OUTPUT> &entry, const std::string &server_name) {
    auto &server_factory = mortred::factory::ServerFactory<mortred::server::BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, [entry]() -> std::unique_ptr<mortred::server::BaseAiServer> {
        mortred::server::CvServerSpec<OUTPUT> spec;
        spec.server_section = entry.server_section;
        spec.model_section = entry.model_section;
        spec.display_name = entry.display_name;
        spec.make_worker = entry.make_worker;
        spec.fill_response = entry.fill_response;
        spec.param_specs = entry.param_specs;
        return std::make_unique<mortred::server::CvModelServer<OUTPUT>>(std::move(spec));
    });
    return server_factory.create(server_name);
}

template <typename OUTPUT>
std::unique_ptr<mortred::server::BaseAiServer> create_server(const std::vector<CvModelEntry<OUTPUT>> &entries,
                                                          const std::string &model_section, const std::string &server_name) {
    const auto *entry = find_entry(entries, model_section);
    if (entry == nullptr) {
        LOG(ERROR) << "model section not found in CV catalog: " << model_section;
        return nullptr;
    }
    return create_server(*entry, server_name);
}

} // namespace cv_catalog
} // namespace factory
} // namespace mortred

#endif // MORTRED_FACTORY_CV_CATALOG_H
