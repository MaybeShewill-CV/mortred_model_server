#ifndef MORTRED_MODEL_SERVER_SCENE_SEGMENTATION_TASK_H
#define MORTRED_MODEL_SERVER_SCENE_SEGMENTATION_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "factory/model_catalog.h"
#include "models/scene_segmentation/bisenetv2.h"
#include "models/scene_segmentation/hrnet_segmentation.h"
#include "models/scene_segmentation/msocrnet.h"
#include "models/scene_segmentation/pp_humanseg.h"

namespace jinq {
namespace factory {
namespace scene_segmentation {

using jinq::models::BaseAiModel;

using jinq::models::scene_segmentation::BiseNetV2;
using jinq::models::scene_segmentation::HRNetSegmentation;
using jinq::models::scene_segmentation::MsOcrNet;
using jinq::models::scene_segmentation::PPHumanSeg;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_bisenetv2_segmentor() {
    return std::make_unique<BiseNetV2<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_pphuman_segmentor() {
    return std::make_unique<PPHumanSeg<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_msocrnet_segmentor() {
    return std::make_unique<MsOcrNet<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_hrnet_segmentor() {
    return std::make_unique<HRNetSegmentation<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::scene_segmentation::std_scene_segmentation_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"BISENETV2", "bisenetv2 segmentation", "BISENETV2_SERVER", &create_bisenetv2_segmentor<ImageInput, Output>,
              &jinq::server::response::fill_scene_segmentation, {}},
        Entry{"PPHUMAN_SEG", "pphuman segmentation", "PPHUMAN_SEG_SERVER", &create_pphuman_segmentor<ImageInput, Output>,
              &jinq::server::response::fill_scene_segmentation, {}},
        Entry{"HRNET", "hrnet segmentation", "HRNET_SERVER", &create_hrnet_segmentor<ImageInput, Output>,
              &jinq::server::response::fill_scene_segmentation, {}},
    };
    return entries;
}

using BenchEntry = jinq::factory::model_catalog::ModelCatalogEntry<ImageInput, Output>;

inline const std::vector<BenchEntry> &bench_catalog() {
    static const std::vector<BenchEntry> entries = {
        BenchEntry{"MSOCRNET", "MsOcrNet scene segmentation", &create_msocrnet_segmentor<ImageInput, Output>},
    };
    return entries;
}

} // namespace scene_segmentation
} // namespace factory
} // namespace jinq

#endif
