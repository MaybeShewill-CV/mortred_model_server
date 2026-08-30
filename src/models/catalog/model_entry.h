#ifndef MORTRED_MODELS_CATALOG_MODEL_ENTRY_H
#define MORTRED_MODELS_CATALOG_MODEL_ENTRY_H

#include <string>

namespace jinq {
namespace models {
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

inline bool model_entry_valid(const ModelEntry &entry) { return !entry.model_section.empty() && !entry.display_name.empty(); }

/***
 * Identity of a model that is mounted on the generic CV server. The extra
 * section points at the server side TOML configuration.
 */
struct ServedModelEntry : public ModelEntry {
    std::string server_section;
};

inline bool served_model_entry_valid(const ServedModelEntry &entry) { return model_entry_valid(entry) && !entry.server_section.empty(); }

} // namespace catalog
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_CATALOG_MODEL_ENTRY_H
