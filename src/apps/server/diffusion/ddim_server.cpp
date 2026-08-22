/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: ddim_server.cpp
* Date: 26-8-23
************************************************/

#include "apps/common/model_server_main.h"
#include "apps/server/diffusion/diffusion_model_adapter.h"
#include "models/diffusion/ddim_sampler.h"

int main(int argc, char** argv) {
    using namespace jinq::apps::diffusion;
    using SamplerInput = jinq::models::io_define::diffusion::std_ddim_input;
    using SamplerOutput = jinq::models::io_define::diffusion::std_ddim_output;
    using Sampler = jinq::models::diffusion::DDIMSampler<SamplerInput, SamplerOutput>;

    return jinq::apps::run_model_server_main(
        argc, argv, "DDIM_SERVER",
        [](const std::string& name)
            -> std::unique_ptr<jinq::server::BaseAiServer> {
            using Adapter = DiffusionModelAdapter<Sampler, SamplerInput, SamplerOutput>;
            using Output = Base64Output;
            jinq::server::CvServerSpec<Output> spec;
            spec.server_section = "DDIM_SERVER";
            spec.model_section = "DDIM";
            spec.display_name = "DDIM diffusion sampler";
            spec.make_worker = [name](const std::string&) {
                return std::make_unique<Adapter>(name);
            };
            return std::make_unique<jinq::server::CvModelServer<Output>>(std::move(spec));
        });
}
