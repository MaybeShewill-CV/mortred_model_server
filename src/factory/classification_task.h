#ifndef MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H
#define MORTRED_MODEL_SERVER_CLASSIFICATION_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/classification/densenet.h"
#include "models/classification/mobilenetv2.h"
#include "models/classification/resnet.h"

namespace jinq {
namespace factory {
namespace classification {

using jinq::models::BaseAiModel;

using jinq::models::classification::DenseNet;
using jinq::models::classification::MobileNetv2;
using jinq::models::classification::ResNet;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_mobilenetv2_classifier(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<MobileNetv2<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_resnet_classifier(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<ResNet<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_densenet_classifier(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<DenseNet<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::classification::std_classification_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

/*** request-overridable classification parameters: top_k keeps the k highest
 * scores (descending) and shrinks the response payload; the full class-index
 * ordered array is the legacy/default behavior */
inline const std::vector<jinq::models::backend::ParamSpec> &classification_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::i32("top_k").range(1, 1000).desc("keep the k highest scores, descending"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"MOBILENETV2", "Mobilenetv2 classification", "MOBILENETV2_CLASSIFICATION_SERVER",
              &create_mobilenetv2_classifier<ImageInput, Output>, &jinq::server::response::fill_classification,
              classification_param_specs()},
        Entry{"RESNET", "Resnet classification", "RESNET_CLASSIFICATION_SERVER", &create_resnet_classifier<ImageInput, Output>,
              &jinq::server::response::fill_classification, classification_param_specs()},
        Entry{"DENSENET", "densenet classification", "DENSENET_CLASSIFICATION_SERVER", &create_densenet_classifier<ImageInput, Output>,
              &jinq::server::response::fill_classification, classification_param_specs()},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_server(const std::string &model_section, const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), model_section, server_name);
}

} // namespace classification
} // namespace factory
} // namespace jinq

#endif
