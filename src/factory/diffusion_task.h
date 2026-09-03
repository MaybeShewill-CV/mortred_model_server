#ifndef MORTRED_MODEL_SERVER_DIFFUSION_TASK_H
#define MORTRED_MODEL_SERVER_DIFFUSION_TASK_H

#include <memory>
#include <string>
#include <vector>

#include "factory/cv_catalog.h"
#include "factory/diffusion_model_adapter.h"
#include "models/diffusion/cls_cond_ddim_sampler.h"
#include "models/diffusion/ddim_sampler.h"
#include "models/diffusion/ddpm_sampler.h"
#include "models/diffusion/ldm_sampler.h"

namespace jinq {
namespace factory {
namespace diffusion {

using jinq::models::BaseAiModel;

using jinq::models::diffusion::ClsCondDDIMSampler;
using jinq::models::diffusion::DDIMSampler;
using jinq::models::diffusion::DDPMSampler;
using jinq::models::diffusion::LDMSampler;

template <typename INPUT, typename OUTPUT> std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_ddpm_sampler(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<DDPMSampler<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT> std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_ddim_sampler(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<DDIMSampler<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_cls_cond_ddim_sampler(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<ClsCondDDIMSampler<INPUT, OUTPUT>>();
}

template <typename INPUT, typename OUTPUT> std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_ldm_sampler(const std::string &model_name) {
    (void)model_name;
    return std::make_unique<LDMSampler<INPUT, OUTPUT>>();
}

using ImageInput = jinq::models::io_define::common_io::image_input;
using Base64Output = jinq::models::io_define::common_io::base64_input;

// every sampler is mounted on the server through the same base64 adapter, so
// the whole task shares one output type and one catalog
template <typename SAMPLER, typename SAMPLER_INPUT, typename SAMPLER_OUTPUT>
std::unique_ptr<BaseAiModel<ImageInput, Base64Output>> make_server_worker(const std::string &worker_name) {
    (void)worker_name;
    return std::make_unique<DiffusionModelAdapter<SAMPLER, SAMPLER_INPUT, SAMPLER_OUTPUT>>();
}

using Entry = jinq::factory::cv_catalog::CvModelEntry<Base64Output>;

/*** request-overridable diffusion knobs, per sampler family. Step counts
 * change the compute time drastically - async callers should size the
 * deadline accordingly (same caveat as SAM AMG points_per_side) */
inline const std::vector<jinq::models::backend::ParamSpec> &ddpm_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::i32("timesteps").range(1, 1000).desc("sampling timesteps (more = slower, higher quality)"),
    };
    return specs;
}

inline const std::vector<jinq::models::backend::ParamSpec> &ddim_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::i32("sample_steps").range(1, 1000).desc("DDIM sampling steps (fewer = faster)"),
        jinq::models::backend::ParamSpec::f32("eta").range(0.0, 1.0).desc("stochasticity (0 = deterministic ODE, 1 = full stochastic)"),
    };
    return specs;
}

inline const std::vector<jinq::models::backend::ParamSpec> &cls_cond_ddim_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::i32("sample_steps").range(1, 1000).desc("DDIM sampling steps (fewer = faster)"),
        jinq::models::backend::ParamSpec::f32("eta").range(0.0, 1.0).desc("stochasticity (0 = deterministic ODE, 1 = full stochastic)"),
        jinq::models::backend::ParamSpec::i32("cls_id").range(0, 9999).desc("conditioning class id (model dependent)"),
    };
    return specs;
}

inline const std::vector<jinq::models::backend::ParamSpec> &ldm_param_specs() {
    static const std::vector<jinq::models::backend::ParamSpec> specs = {
        jinq::models::backend::ParamSpec::i32("step_size").range(1, 1000).desc("latent sampler steps (fewer = faster)"),
    };
    return specs;
}

inline const std::vector<Entry> &catalog() {
    using jinq::models::io_define::diffusion::std_cls_cond_ddim_input;
    using jinq::models::io_define::diffusion::std_cls_cond_ddim_output;
    using jinq::models::io_define::diffusion::std_ddim_input;
    using jinq::models::io_define::diffusion::std_ddim_output;
    using jinq::models::io_define::diffusion::std_ddpm_input;
    using jinq::models::io_define::diffusion::std_ddpm_output;
    using jinq::models::io_define::diffusion::std_ldm_input;
    using jinq::models::io_define::diffusion::std_ldm_output;

    using ClsCondDDIM = ClsCondDDIMSampler<std_cls_cond_ddim_input, std_cls_cond_ddim_output>;
    using DDIM = DDIMSampler<std_ddim_input, std_ddim_output>;
    using DDPM = DDPMSampler<std_ddpm_input, std_ddpm_output>;
    using LDM = LDMSampler<std_ldm_input, std_ldm_output>;

    static const std::vector<Entry> entries = {
        Entry{"DDPM", "DDPM diffusion sampler", "DDPM_SERVER", &make_server_worker<DDPM, std_ddpm_input, std_ddpm_output>,
              &jinq::server::response::fill_base64_image, ddpm_param_specs()},
        Entry{"DDIM", "DDIM diffusion sampler", "DDIM_SERVER", &make_server_worker<DDIM, std_ddim_input, std_ddim_output>,
              &jinq::server::response::fill_base64_image, ddim_param_specs()},
        Entry{"CLS_COND_DDIM", "class conditional DDIM sampler", "CLS_COND_DDIM_SERVER",
              &make_server_worker<ClsCondDDIM, std_cls_cond_ddim_input, std_cls_cond_ddim_output>,
              &jinq::server::response::fill_base64_image, cls_cond_ddim_param_specs()},
        Entry{"LDM", "latent diffusion sampler", "LDM_SERVER", &make_server_worker<LDM, std_ldm_input, std_ldm_output>,
              &jinq::server::response::fill_base64_image, ldm_param_specs()},
    };
    return entries;
}

inline std::unique_ptr<jinq::server::BaseAiServer> create_server(const std::string &model_section, const std::string &server_name) {
    return jinq::factory::cv_catalog::create_server(catalog(), model_section, server_name);
}

} // namespace diffusion
} // namespace factory
} // namespace jinq

#endif
