/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: pp_matting_server.cpp
* Date: 2026-08-19
************************************************/

// pp_matting server tool

#include "apps/common/model_server_main.h"
#include "factory/matting_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "PP_MATTING_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::matting::create_pp_matting_server(server_name);
        });
}
