/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: attentive_gan_derain_server.cpp
 * Date: 26-8-19
 ************************************************/

// attentive_gan_derain server tool

#include "apps/common/model_server_main.h"
#include "factory/enhancement_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "ATTENTIVE_GAN_DERAIN_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::enhancement::create_attentivegan_derain_server(server_name);
        });
}
