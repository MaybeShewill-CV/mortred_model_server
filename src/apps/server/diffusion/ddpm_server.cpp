/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: ddpm_server.cpp
* Date: 26-8-23
************************************************/

#include "apps/common/model_server_main.h"
#include "apps/server/diffusion/diffusion_model_adapter.h"
#include "models/diffusion/ddpm_sampler.h"

int main(int argc, char** argv) {
    using namespace jinq::apps::diffusion;
    using SamplerInput = jinq::models::io_define::diffusion::std_ddpm_input;
    using SamplerOutput = jinq::models::io_define::diffusion::std_ddpm_output;
    using Sampler = jinq::models::diffusion::DDPMSampler<SamplerInput, SamplerOutput>;

    return jinq::apps::run_model_server_main(
        argc, argv, "DDPM_SERVER",
        [](const std::string& name)
            -> std::unique_ptr<jinq::server::BaseAiServer> {
            // use the generic CV server with the adapter as worker
            // (the adapter converts base64_input to the sampler's contract)
            using Adapter = DiffusionModelAdapter<Sampler, SamplerInput, SamplerOutput>;
            using Output = Base64Output;
            jinq::server::CvServerSpec<Output> spec;
            spec.server_section = "DDPM_SERVER";
            spec.model_section = "DDPM";
            spec.display_name = "DDPM diffusion sampler";
            spec.make_worker = [name](const std::string&) {
                return std::make_unique<Adapter>(name);
            };
            return std::make_unique<jinq::server::CvModelServer<Output>>(std::move(spec));
        });
}
