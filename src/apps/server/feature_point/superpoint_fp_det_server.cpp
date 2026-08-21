/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: superpoint_fp_det_server.cpp
 * Date: 26-8-19
 ************************************************/

// superpoint_fp_det server tool

#include "apps/common/model_server_main.h"
#include "factory/feature_point_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "SUPERPOINT_FP_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::feature_point::create_superpoint_fp_server(server_name);
        });
}
