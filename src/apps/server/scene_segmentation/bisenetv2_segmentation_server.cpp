/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: bisenetv2_segmentation_server.cpp
* Date: 2026-08-19
************************************************/

// bisenetv2_segmentation server tool

#include "apps/common/model_server_main.h"
#include "factory/scene_segmentation_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "BISENETV2_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::scene_segmentation::create_bisenetv2_server(server_name);
        });
}
