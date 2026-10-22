#ifndef MORTRED_FACTORY_MODEL_CATALOG_H
#define MORTRED_FACTORY_MODEL_CATALOG_H

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "models/base_model.h"

namespace mortred {
namespace factory {

/***
 * Catalog entry identity shared by every catalog entry: the TOML section that
 * owns the model configuration plus a human readable name for logs and the
 * web console. Typed task catalogs derive from this struct.
 */
namespace catalog {

/***
 * Task-agnostic identity shared by every catalog entry: the TOML section that
 * owns the model configuration plus a human readable name for logs and the
 * web console. Typed task catalogs derive from this struct.
 */
struct ModelEntry {
    std::string model_section;
    std::string display_name;
};

inline bool model_entry_valid(const ModelEntry &entry) {
    return !entry.model_section.empty() && !entry.display_name.empty();
}

/***
 * Identity of a model that is mounted on the generic CV server. The extra
 * section points at the server side TOML configuration.
 */
struct ServedModelEntry : public ModelEntry {
    std::string server_section;
};

inline bool served_model_entry_valid(const ServedModelEntry &entry) {
    return model_entry_valid(entry) && !entry.server_section.empty();
}

}  // namespace catalog

namespace model_catalog {

/***
 * Catalog entry for model families that are consumed directly by benchmarks
 * and in-process callers instead of being mounted on the generic CV server.
 * Keeping it separate from CvModelEntry avoids forcing a server section onto
 * models that have no HTTP surface yet.
 */
template <typename INPUT, typename OUTPUT> struct ModelCatalogEntry : public catalog::ModelEntry {
    std::function<std::unique_ptr<mortred::models::BaseAiModel<INPUT, OUTPUT>>()> make_model;
};

template <typename INPUT, typename OUTPUT>
const ModelCatalogEntry<INPUT, OUTPUT> *find_entry(const std::vector<ModelCatalogEntry<INPUT, OUTPUT>> &entries,
                                                   const std::string &model_section) {
    const auto found = std::find_if(entries.begin(), entries.end(), [&model_section](const ModelCatalogEntry<INPUT, OUTPUT> &entry) {
        return entry.model_section == model_section;
    });
    return found == entries.end() ? nullptr : &*found;
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<mortred::models::BaseAiModel<INPUT, OUTPUT>> create_model(const std::vector<ModelCatalogEntry<INPUT, OUTPUT>> &entries,
                                                                       const std::string &model_section) {
    const auto *entry = find_entry(entries, model_section);
    if (entry == nullptr || !entry->make_model) {
        LOG(ERROR) << "model section not found in model catalog: " << model_section;
        return nullptr;
    }
    return entry->make_model();
}

} // namespace model_catalog
} // namespace factory
} // namespace mortred

#endif // MORTRED_FACTORY_MODEL_CATALOG_H
