#ifndef MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H
#define MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/enhancement/attentive_gan_derain_net.h"
#include "models/enhancement/enlightengan.h"
#include "models/enhancement/real_esrgan.h"

namespace jinq {
namespace factory {
namespace enhancement {

using jinq::models::BaseAiModel;

using jinq::models::enhancement::AttentiveGanDerain;
using jinq::models::enhancement::EnlightenGan;
using jinq::models::enhancement::RealEsrGan;

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_enlightengan_enhancementor() {
    return std::make_unique<EnlightenGan<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_attentivegan_enhancementor() {
    return std::make_unique<AttentiveGanDerain<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_realesrgan_enhancementor() {
    return std::make_unique<RealEsrGan<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::enhancement::std_enhancement_output;
using jinq::server::ImageInput;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"ENLIGHTEN_GAN", "enlighten gan", "ENLIGHTEN_GAN_SERVER", &create_enlightengan_enhancementor<ImageInput, Output>,
              &jinq::server::response::fill_enhancement, {}},
        Entry{"ATTENTIVE_GAN_DERAIN", "attentive gan derain", "ATTENTIVE_GAN_DERAIN_SERVER",
              &create_attentivegan_enhancementor<ImageInput, Output>, &jinq::server::response::fill_enhancement, {}},
        Entry{"REAL_ESRGAN", "real esr-gan", "REAL_ESRGAN_SERVER", &create_realesrgan_enhancementor<ImageInput, Output>,
              &jinq::server::response::fill_enhancement, {}},
    };
    return entries;
}

} // namespace enhancement
} // namespace factory
} // namespace jinq

#endif
