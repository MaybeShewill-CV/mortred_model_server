#ifndef MORTRED_MODEL_SERVER_SAM_TASK_H
#define MORTRED_MODEL_SERVER_SAM_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "factory/model_catalog.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "models/segment_anything/fast_sam/fast_sam_segmentor.h"
#include "models/segment_anything/sam_automask_generator/sam_automask_generator.h"
#include "models/segment_anything/sam_prediction/sam_predictor.h"

namespace jinq {
namespace factory {
namespace segment_anything {

using jinq::models::BaseAiModel;

using jinq::models::segment_anything::FastSamSegmentor;
using jinq::models::segment_anything::SamAutoMaskGenerator;
using jinq::models::segment_anything::SamPredictor;

template <typename INPUT, typename OUTPUT> std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_sam_predictor(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<SamPredictor<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_sam_auto_mask_generator(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<SamAutoMaskGenerator<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_fast_sam_segmentor(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<FastSamSegmentor<INPUT, OUTPUT>>();
}

// SAM families have distinct IO contracts: the prompt predictor and fast-sam
// are consumed directly by benchmarks, while the auto mask generator is served
// over HTTP. Each family therefore keeps its own typed catalog.
using MatInput = jinq::models::io_define::common_io::mat_input;
using PromptInput = jinq::models::io_define::segment_anything::sam_prompt_input;
using PromptOutput = jinq::models::io_define::segment_anything::std_sam_prompt_output;
using FastSamOutput = jinq::models::io_define::segment_anything::std_fast_sam_output;
using AmgOutput = jinq::models::io_define::segment_anything::std_sam_amg_output;
using jinq::server::ImageInput;

using PredictorEntry = jinq::factory::model_catalog::ModelCatalogEntry<PromptInput, PromptOutput>;
using FastSamEntry = jinq::factory::model_catalog::ModelCatalogEntry<MatInput, FastSamOutput>;
using AmgEntry = jinq::factory::cv_catalog::CvModelEntry<AmgOutput>;

/*** request-overridable SAM AMG parameters. NOTE: points_per_side changes
 * the compute time drastically (it is the prompt grid density) - requests on
 * async long tasks should account for the longer deadline */
inline const std::vector<jinq::models::backend::ParamSpec> &sam_amg_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::i32("points_per_side").range(1, 64).desc("prompt point grid density (n x n points)"),
        jinq::models::backend::ParamSpec::f32("pred_iou_thresh").range(0.0, 1.0).desc("mask quality filter"),
        jinq::models::backend::ParamSpec::f32("stability_score_thresh").range(0.0, 1.0).desc("mask stability filter"),
        jinq::models::backend::ParamSpec::f32("box_nms_thresh").range(0.0, 1.0).desc("mask NMS IoU threshold"),
        jinq::models::backend::ParamSpec::i32("min_mask_region_area").range(0, 100000).desc("drop masks smaller than this many pixels"),
    };
    return specs;
}

inline const std::vector<PredictorEntry> &predictor_catalog() {
    static const std::vector<PredictorEntry> entries = {
        PredictorEntry{"SAM_PREDICTOR", "SAM promptable segmentation", &create_sam_predictor<PromptInput, PromptOutput>},
    };
    return entries;
}

inline const std::vector<FastSamEntry> &fast_sam_catalog() {
    static const std::vector<FastSamEntry> entries = {
        FastSamEntry{"FAST_SAM", "FastSAM everything segmentation", &create_fast_sam_segmentor<MatInput, FastSamOutput>},
    };
    return entries;
}

inline const std::vector<AmgEntry> &amg_catalog() {
    static const std::vector<AmgEntry> entries = {
        AmgEntry{"SAM_AMG", "SAM automatic mask generator", "SAM_AMG_SERVER", &create_sam_auto_mask_generator<ImageInput, AmgOutput>,
                 &jinq::server::response::fill_sam_amg, sam_amg_param_specs()},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_amg_server(const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(amg_catalog(), "SAM_AMG", server_name);
}

} // namespace segment_anything
} // namespace factory
} // namespace jinq

#endif
