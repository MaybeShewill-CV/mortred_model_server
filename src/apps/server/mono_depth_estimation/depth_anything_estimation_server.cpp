/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: depth_anything_estimation_server.cpp
* Date: 2026-08-19
************************************************/

// depth_anything_estimation server tool

#include "apps/common/model_server_main.h"
#include "factory/mono_depth_estimate_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "DEPTH_ANYTHING_ESTIMATION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::mono_depth_estimation::create_depth_anything_estimation_server(server_name);
        });
}
