#ifndef MORTRED_MODEL_SERVER_MATTING_TASK_H
#define MORTRED_MODEL_SERVER_MATTING_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/matting/modnet_matting.h"
#include "models/matting/pp_matting.h"

namespace mortred {
namespace factory {
namespace matting {

using mortred::models::BaseAiModel;

using mortred::models::matting::ModNetMatting;
using mortred::models::matting::PPMatting;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_modnet_segmentor() {
    return std::make_unique<ModNetMatting<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_ppmatting_segmentor() {
    return std::make_unique<PPMatting<INPUT, OUTPUT>>();
}

using Output = mortred::models::io_define::matting::std_matting_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"modnet", "MODNet matting", "MODNET_SERVER", &create_modnet_segmentor<ImageInput, Output>,
              &mortred::server::response::fill_matting, {}, "modnet"},
        Entry{"ppmatting_512", "PP-Matting 512", "PP_MATTING_SERVER", &create_ppmatting_segmentor<ImageInput, Output>,
              &mortred::server::response::fill_matting, {}, "ppmatting"},
        Entry{"ppmatting_1024", "PP-Matting 1024", "PP_MATTING_SERVER", &create_ppmatting_segmentor<ImageInput, Output>,
              &mortred::server::response::fill_matting, {}, "ppmatting"},
        Entry{"ppmatting_resnet34", "PP-Matting ResNet34", "PP_MATTING_SERVER", &create_ppmatting_segmentor<ImageInput, Output>,
              &mortred::server::response::fill_matting, {}, "ppmatting"},
        Entry{"ppmatting_v2", "PP-Matting v2", "PP_MATTING_SERVER", &create_ppmatting_segmentor<ImageInput, Output>,
              &mortred::server::response::fill_matting, {}, "ppmatting"},
    };
    return entries;
}

} // namespace matting
} // namespace factory
} // namespace mortred

#endif
