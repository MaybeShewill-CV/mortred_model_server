/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: feature_embedding_task.h
 * Date: 26-10-6
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_FEATURE_EMBEDDING_TASK_H
#define MORTRED_MODEL_SERVER_FEATURE_EMBEDDING_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/feature_embedding/dinov2.h"
#include "models/model_io_define.h"

namespace mortred {
namespace factory {
namespace feature_embedding {

using mortred::models::BaseAiModel;

using mortred::models::feature_embedding::Dinov2;

using Output = mortred::models::io_define::feature_embedding::std_feature_embedding_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable feature embedding parameters. NOTE: normalize L2-
 * normalizes the returned embedding (unit hypersphere); it changes the vector
 * scale but never the ordering of cosine-similarity comparisons. pooling is a
 * STRING enum: the current exports carry only the [CLS] token, so "cls" is the
 * only accepted value today. The model side already understands all-token
 * exports ([1,T,D]): once a re-export is deployed, adding "mean" to the enum
 * below is the ONLY change needed - no model code edits. */
inline const std::vector<mortred::models::backend::ParamSpec> &feature_embedding_param_specs() {
    static const std::vector<mortred::models::backend::ParamSpec> specs = {
        mortred::models::backend::ParamSpec::boolean("normalize").desc("L2-normalize the returned embedding"),
        mortred::models::backend::ParamSpec::str("pooling")
            .values({"cls"})
            .desc("token pooling strategy: cls = [CLS] token (default); mean (all-token average) is enabled once an "
                  "all-token export is deployed"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"dinov2_vits14", "DINOv2 ViT-S/14 feature embedding", "DINOV2_FEATURE_EMBEDDING_SERVER",
              [] { return std::make_unique<Dinov2<ImageInput, Output>>(); }, &mortred::server::response::fill_feature_embedding,
              feature_embedding_param_specs(), "dinov2"},
        Entry{"dinov2_vitb14", "DINOv2 ViT-B/14 feature embedding", "DINOV2_FEATURE_EMBEDDING_SERVER",
              &create_dinov2_feature_extractor<ImageInput, Output>, &mortred::server::response::fill_feature_embedding,
              feature_embedding_param_specs(), "dinov2"},
        Entry{"dinov2_vitl14", "DINOv2 ViT-L/14 feature embedding", "DINOV2_FEATURE_EMBEDDING_SERVER",
              &create_dinov2_feature_extractor<ImageInput, Output>, &mortred::server::response::fill_feature_embedding,
              feature_embedding_param_specs(), "dinov2"},
    };
    return entries;
}

} // namespace feature_embedding
} // namespace factory
} // namespace mortred

#endif // MORTRED_MODEL_SERVER_FEATURE_EMBEDDING_TASK_H
