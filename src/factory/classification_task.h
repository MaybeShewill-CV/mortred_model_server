#ifndef MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H
#define MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/classification/densenet.h"
#include "models/classification/mobilenetv2.h"
#include "models/classification/resnet.h"

namespace mortred {
namespace factory {
namespace classification {

using mortred::models::BaseAiModel;

using mortred::models::classification::DenseNet;
using mortred::models::classification::MobileNetv2;
using mortred::models::classification::ResNet;

using Output = mortred::models::io_define::classification::std_classification_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable classification parameters: top_k keeps the k highest
 * scores (descending) and shrinks the response payload; the full class-index
 * ordered array is the legacy/default behavior */
inline const std::vector<mortred::models::backend::ParamSpec> &classification_param_specs() {
    static const std::vector<mortred::models::backend::ParamSpec> specs = {
        mortred::models::backend::ParamSpec::i32("top_k").range(1, 1000).desc("keep the k highest scores, descending"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"mobilenetv2", "Mobilenetv2 classification", "MOBILENETV2_CLASSIFICATION_SERVER",
              [] { return std::make_unique<MobileNetv2<ImageInput, Output>>(); }, &mortred::server::response::fill_classification,
              classification_param_specs(), "mobilenetv2"},
        Entry{"resnet", "Resnet classification", "RESNET_CLASSIFICATION_SERVER", [] { return std::make_unique<ResNet<ImageInput, Output>>(); },
              &mortred::server::response::fill_classification, classification_param_specs(), "resnet"},
        Entry{"densenet", "Densenet classification", "DENSENET_CLASSIFICATION_SERVER", [] { return std::make_unique<DenseNet<ImageInput, Output>>(); },
              &mortred::server::response::fill_classification, classification_param_specs(), "densenet"},
    };
    return entries;
}

} // namespace classification
} // namespace factory
} // namespace mortred

#endif
