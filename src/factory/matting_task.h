#ifndef MORTRED_MODEL_SERVER_MATTING_TASK_H
#define MORTRED_MODEL_SERVER_MATTING_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/matting/modnet_matting.h"
#include "models/matting/pp_matting.h"

namespace jinq {
namespace factory {
namespace matting {

using jinq::models::BaseAiModel;

using jinq::models::matting::ModNetMatting;
using jinq::models::matting::PPMatting;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_modnet_segmentor(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<ModNetMatting<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_ppmatting_segmentor(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<PPMatting<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::matting::std_matting_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"PP_MATTING", "pp matting", "PP_MATTING_SERVER", &create_ppmatting_segmentor<ImageInput, Output>,
              &jinq::server::response::fill_matting, {}},
        Entry{"MODNET", "modnet", "MODNET_SERVER", &create_modnet_segmentor<ImageInput, Output>, &jinq::server::response::fill_matting, {}},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_server(const std::string &model_section, const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), model_section, server_name);
}

} // namespace matting
} // namespace factory
} // namespace jinq

#endif
