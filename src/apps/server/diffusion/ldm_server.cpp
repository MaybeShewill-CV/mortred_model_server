/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: ldm_server.cpp
* Date: 26-8-23
************************************************/

#include "apps/common/model_server_main.h"
#include "apps/server/diffusion/diffusion_model_adapter.h"
#include "models/diffusion/ldm_sampler.h"

int main(int argc, char** argv) {
    using namespace jinq::apps::diffusion;
    using SamplerInput = jinq::models::io_define::diffusion::std_ldm_input;
    using SamplerOutput = jinq::models::io_define::diffusion::std_ldm_output;
    using Sampler = jinq::models::diffusion::LDMSampler<SamplerInput, SamplerOutput>;

    return jinq::apps::run_model_server_main(
        argc, argv, "LDM_SERVER",
        [](const std::string& name)
            -> std::unique_ptr<jinq::server::BaseAiServer> {
            using Adapter = DiffusionModelAdapter<Sampler, SamplerInput, SamplerOutput>;
            using Output = Base64Output;
            jinq::server::CvServerSpec<Output> spec;
            spec.server_section = "LDM_SERVER";
            spec.model_section = "LDM";
            spec.display_name = "Latent diffusion sampler";
            spec.make_worker = [name](const std::string&) {
                return std::make_unique<Adapter>(name);
            };
            return std::make_unique<jinq::server::CvModelServer<Output>>(std::move(spec));
        });
}
