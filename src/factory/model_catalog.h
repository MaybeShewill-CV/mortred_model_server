#ifndef MORTRED_FACTORY_MODEL_CATALOG_H
#define MORTRED_FACTORY_MODEL_CATALOG_H

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "models/base_model.h"
#include "models/catalog/model_entry.h"

namespace jinq {
namespace factory {
namespace model_catalog {

/***
 * Catalog entry for model families that are consumed directly by benchmarks
 * and in-process callers instead of being mounted on the generic CV server. Keeping it
 * separate from CvModelEntry avoids forcing a server section onto models that
 * have no HTTP surface yet.
 */
template <typename INPUT, typename OUTPUT> struct ModelCatalogEntry : public jinq::models::catalog::ModelEntry {
    std::function<std::unique_ptr<jinq::models::BaseAiModel<INPUT, OUTPUT>>()> make_model;
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
std::unique_ptr<jinq::models::BaseAiModel<INPUT, OUTPUT>> create_model(const std::vector<ModelCatalogEntry<INPUT, OUTPUT>> &entries,
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
} // namespace jinq

#endif // MORTRED_FACTORY_MODEL_CATALOG_H
