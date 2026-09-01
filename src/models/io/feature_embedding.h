/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: feature_embedding.h
 * Date: 26-10-6
 ************************************************/

#ifndef MORTRED_MODELS_IO_FEATURE_EMBEDDING_H
#define MORTRED_MODELS_IO_FEATURE_EMBEDDING_H

#include <vector>

namespace jinq {
namespace models {
namespace io_define {
namespace feature_embedding {

// image feature embedding (e.g. the DINOv2 ViT [CLS] token output)

struct feature_embedding_output {
    std::vector<float> embedding;
};
using std_feature_embedding_output = feature_embedding_output;

} // namespace feature_embedding
} // namespace io_define
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_IO_FEATURE_EMBEDDING_H
