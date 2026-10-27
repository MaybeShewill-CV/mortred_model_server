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

namespace mortred {
namespace factory {
namespace scene_segmentation {

using mortred::models::BaseAiModel;

using mortred::models::scene_segmentation::BiseNetV2;
using mortred::models::scene_segmentation::HRNetSegmentation;
using mortred::models::scene_segmentation::MsOcrNet;
using mortred::models::scene_segmentation::PPHumanSeg;

using Output = mortred::models::io_define::scene_segmentation::std_scene_segmentation_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"bisenetv2", "BiSeNetV2 segmentation", "BISENETV2_SERVER", [] { return std::make_unique<BiseNetV2<ImageInput, Output>>(); },
              &mortred::server::response::fill_scene_segmentation, {}, "bisenetv2"},
        Entry{"pphuman_mobile", "PP-HumanSeg mobile segmentation", "PPHUMAN_SEG_SERVER",
              [] { return std::make_unique<PPHumanSeg<ImageInput, Output>>(); }, &mortred::server::response::fill_scene_segmentation, {}, "pphuman"},
        Entry{"pphuman_lite", "PP-HumanSeg lite segmentation", "PPHUMAN_SEG_SERVER", [] { return std::make_unique<PPHumanSeg<ImageInput, Output>>(); },
              &mortred::server::response::fill_scene_segmentation, {}, "pphuman"},
        Entry{"pphuman_server", "PP-HumanSeg server segmentation", "PPHUMAN_SEG_SERVER",
              [] { return std::make_unique<PPHumanSeg<ImageInput, Output>>(); }, &mortred::server::response::fill_scene_segmentation, {}, "pphuman"},
        Entry{"hrnet", "HRNet segmentation", "HRNET_SERVER", [] { return std::make_unique<HRNetSegmentation<ImageInput, Output>>(); },
              &mortred::server::response::fill_scene_segmentation, {}, "hrnet"},
        Entry{"msocrnet", "MSOCRNet segmentation", "MSOCRNET_SERVER", [] { return std::make_unique<MsOcrNet<ImageInput, Output>>(); },
              &mortred::server::response::fill_scene_segmentation, {}, "msocrnet"},
    };
    return entries;
}

using BenchEntry = mortred::factory::model_catalog::ModelCatalogEntry<ImageInput, Output>;

inline const std::vector<BenchEntry> &bench_catalog() {
    static const std::vector<BenchEntry> entries;
    return entries;
}

} // namespace scene_segmentation
} // namespace factory
} // namespace mortred

#endif
