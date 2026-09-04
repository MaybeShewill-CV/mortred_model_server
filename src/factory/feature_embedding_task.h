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

namespace jinq {
namespace factory {
namespace feature_embedding {

using jinq::models::BaseAiModel;

using jinq::models::feature_embedding::Dinov2;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_dinov2_feature_extractor() {
    return std::make_unique<Dinov2<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::feature_embedding::std_feature_embedding_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable feature embedding parameters. NOTE: normalize L2-
 * normalizes the returned embedding (unit hypersphere); it changes the vector
 * scale but never the ordering of cosine-similarity comparisons. pooling is a
 * STRING enum: the current exports carry only the [CLS] token, so "cls" is the
 * only accepted value today. The model side already understands all-token
 * exports ([1,T,D]): once a re-export is deployed, adding "mean" to the enum
 * below is the ONLY change needed - no model code edits. */
inline const std::vector<jinq::models::backend::ParamSpec> &feature_embedding_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::boolean("normalize").desc("L2-normalize the returned embedding"),
        jinq::models::backend::ParamSpec::str("pooling")
            .values({"cls"})
            .desc("token pooling strategy: cls = [CLS] token (default); mean (all-token average) is enabled once an "
                  "all-token export is deployed"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"DINOV2", "DINOv2 image feature embedding", "DINOV2_FEATURE_EMBEDDING_SERVER",
              &create_dinov2_feature_extractor<ImageInput, Output>, &jinq::server::response::fill_feature_embedding,
              feature_embedding_param_specs()},
    };
    return entries;
}

} // namespace feature_embedding
} // namespace factory
} // namespace jinq

#endif // MORTRED_MODEL_SERVER_FEATURE_EMBEDDING_TASK_H
