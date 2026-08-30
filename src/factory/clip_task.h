#ifndef MORTRED_MODEL_SERVER_CLIP_TASK_H
#define MORTRED_MODEL_SERVER_CLIP_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/model_catalog.h"
#include "models/base_model.h"
#include "models/clip/openai_clip.h"
#include "models/model_io_define.h"

namespace jinq {
namespace factory {
namespace clip {

using jinq::models::BaseAiModel;

using jinq::models::clip::OpenAiClip;

template <typename INPUT, typename OUTPUT> std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_openai_clip(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<OpenAiClip<INPUT, OUTPUT>>();
}

// CLIP is consumed directly by benchmarks and pipelines; it has no generic CV
// server surface yet, so its catalog entry carries only model identity.
using ClipInput = jinq::models::io_define::clip::clip_input;
using ClipOutput = jinq::models::io_define::clip::clip_output;
using Entry = jinq::factory::model_catalog::ModelCatalogEntry<ClipInput, ClipOutput>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"OPENAI_CLIP", "OpenAI CLIP vision-language embedding", &create_openai_clip<ClipInput, ClipOutput>},
    };
    return entries;
}

inline std::unique_ptr<BaseAiModel<ClipInput, ClipOutput>> create_model(const std::string &model_section) {
    return jinq::factory::model_catalog::create_model(catalog(), model_section);
}

} // namespace clip
} // namespace factory
} // namespace jinq

#endif
