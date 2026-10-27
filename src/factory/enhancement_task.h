#ifndef MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H
#define MORTRED_MODEL_SERVER_ENHANCEMENT_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "models/enhancement/attentive_gan_derain_net.h"
#include "models/enhancement/enlightengan.h"
#include "models/enhancement/real_esrgan.h"

namespace mortred {
namespace factory {
namespace enhancement {

using mortred::models::BaseAiModel;

using mortred::models::enhancement::AttentiveGanDerain;
using mortred::models::enhancement::EnlightenGan;
using mortred::models::enhancement::RealEsrGan;

using Output = mortred::models::io_define::enhancement::std_enhancement_output;
using mortred::server::ImageInput;
using Entry = mortred::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"attentive_gan", "AttentiveGAN derain", "ATTENTIVE_GAN_DERAIN_SERVER",
              [] { return std::make_unique<AttentiveGanDerain<ImageInput, Output>>(); }, &mortred::server::response::fill_enhancement, {},
              "attentive_gan"},
        Entry{"enlightengan", "EnlightenGAN", "ENLIGHTEN_GAN_SERVER", [] { return std::make_unique<EnlightenGan<ImageInput, Output>>(); },
              &mortred::server::response::fill_enhancement, {}, "enlightengan"},
        Entry{"realesrgan", "Real-ESRGAN", "REAL_ESRGAN_SERVER", [] { return std::make_unique<RealEsrGan<ImageInput, Output>>(); },
              &mortred::server::response::fill_enhancement, {}, "realesrgan"},
    };
    return entries;
}

} // namespace enhancement
} // namespace factory
} // namespace mortred

#endif
