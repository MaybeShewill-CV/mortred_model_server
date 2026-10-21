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

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_mobilenetv2_classifier() {
    return std::make_unique<MobileNetv2<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_resnet_classifier() {
    return std::make_unique<ResNet<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_densenet_classifier() {
    return std::make_unique<DenseNet<INPUT, OUTPUT>>();
}

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
              &create_mobilenetv2_classifier<ImageInput, Output>, &mortred::server::response::fill_classification,
              classification_param_specs(), "mobilenetv2"},
        Entry{"resnet", "Resnet classification", "RESNET_CLASSIFICATION_SERVER", &create_resnet_classifier<ImageInput, Output>,
              &mortred::server::response::fill_classification, classification_param_specs(), "resnet"},
        Entry{"densenet", "Densenet classification", "DENSENET_CLASSIFICATION_SERVER", &create_densenet_classifier<ImageInput, Output>,
              &mortred::server::response::fill_classification, classification_param_specs(), "densenet"},
    };
    return entries;
}

} // namespace classification
} // namespace factory
} // namespace mortred

#endif
