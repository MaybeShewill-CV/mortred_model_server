/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: enlighten_gan_server.cpp
 * Date: 26-8-19
 ************************************************/

// enlighten_gan server tool

#include "apps/common/model_server_main.h"
#include "factory/enhancement_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "ENLIGHTEN_GAN_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::enhancement::create_enlightengan_server(server_name);
        });
}
