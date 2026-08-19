/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: resnet_classification_server.cpp
* Date: 2026-08-19
************************************************/

// resnet_classification server tool

#include "apps/common/model_server_main.h"
#include "factory/classification_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "RESNET_CLASSIFICATION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::classification::create_resnet_cls_server(server_name);
        });
}
