/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: metric3d_estimation_server.cpp
 * Date: 26-8-19
 ************************************************/

// metric3d_estimation server tool

#include "apps/common/model_server_main.h"
#include "factory/mono_depth_estimate_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "METRIC3D_ESTIMATION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::mono_depth_estimation::create_metric3d_estimation_server(server_name);
        });
}
