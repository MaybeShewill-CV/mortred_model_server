/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: sam_amg_server.cpp
* Date: 26-8-23
************************************************/

#include "apps/common/model_server_main.h"
#include "factory/sam_task.h"
#include "models/model_io_define.h"
#include "server/generic_cv_server.h"

int main(int argc, char** argv) {
    using Base64Input = jinq::models::io_define::common_io::base64_input;
    using SamAmgOutput = jinq::models::io_define::segment_anything::std_sam_amg_output;

    return jinq::apps::run_model_server_main(
        argc, argv, "SAM_AMG_SERVER",
        [](const std::string& server_name)
            -> std::unique_ptr<jinq::server::BaseAiServer> {
            jinq::server::CvServerSpec<SamAmgOutput> spec;
            spec.server_section = "SAM_AMG_SERVER";
            spec.model_section = "SAM_AMG";
            spec.display_name = "SAM automatic mask generator";
            spec.make_worker = [server_name](const std::string&) {
                return jinq::factory::segment_anything::create_sam_auto_mask_generator<
                    Base64Input, SamAmgOutput>(server_name);
            };
            return std::make_unique<jinq::server::CvModelServer<SamAmgOutput>>(std::move(spec));
        });
}
