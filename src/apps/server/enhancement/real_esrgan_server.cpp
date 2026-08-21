/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: real_esrgan_server.cpp
 * Date: 26-8-19
 ************************************************/

// real_esrgan server tool

#include "apps/common/model_server_main.h"
#include "factory/enhancement_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "REAL_ESRGAN_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::enhancement::create_realesrgan_server(server_name);
        });
}
