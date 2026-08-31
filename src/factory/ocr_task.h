#ifndef MORTRED_MODEL_SERVER_OCR_TASK_H
#define MORTRED_MODEL_SERVER_OCR_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/ocr/db_text_detector.h"

namespace jinq {
namespace factory {
namespace ocr {

using jinq::models::BaseAiModel;

using jinq::models::ocr::DBTextDetector;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_dbtext_detector(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<DBTextDetector<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::ocr::std_text_regions_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"DBNET", "dbnet", "DBNET_SERVER", &create_dbtext_detector<ImageInput, Output>, &jinq::server::response::fill_text_regions, {}},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_server(const std::string &model_section, const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), model_section, server_name);
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_dbtext_detection_server(const std::string &server_name) {
    return create_server("DBNET", server_name);
}

} // namespace ocr
} // namespace factory
} // namespace jinq

#endif
