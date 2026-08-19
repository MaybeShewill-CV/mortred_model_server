/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: modnet_server.cpp
* Date: 2026-08-19
************************************************/

// modnet server tool

#include "apps/common/model_server_main.h"
#include "factory/matting_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "MODNET_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::matting::create_modnet_server(server_name);
        });
}
