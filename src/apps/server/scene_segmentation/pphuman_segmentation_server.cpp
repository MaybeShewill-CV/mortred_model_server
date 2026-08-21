/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: pphuman_segmentation_server.cpp
 * Date: 26-8-19
 ************************************************/

// pphuman_segmentation server tool

#include "apps/common/model_server_main.h"
#include "factory/scene_segmentation_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "PPHUMAN_SEG_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::scene_segmentation::create_pphuman_seg_server(server_name);
        });
}
