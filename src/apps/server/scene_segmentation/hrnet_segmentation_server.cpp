/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: hrnet_segmentation_server.cpp
 * Date: 26-8-19
 ************************************************/

// hrnet_segmentation server tool

#include "apps/common/model_server_main.h"
#include "factory/scene_segmentation_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "HRNET_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::scene_segmentation::create_hrnet_server(server_name);
        });
}
