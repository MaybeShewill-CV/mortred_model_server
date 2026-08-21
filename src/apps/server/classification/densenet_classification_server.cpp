/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: densenet_classification_server.cpp
 * Date: 26-8-19
 ************************************************/

// densenet_classification server tool

#include "apps/common/model_server_main.h"
#include "factory/classification_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "DENSENET_CLASSIFICATION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::classification::create_densenet_cls_server(server_name);
        });
}
