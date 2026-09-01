/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dinov2_feature_embedding_server.cpp
 * Date: 26-10-6
 ************************************************/

// dinov2 feature embedding server tool

#include "apps/common/model_server_main.h"
#include "factory/feature_embedding_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "DINOV2_FEATURE_EMBEDDING_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::feature_embedding::create_dinov2_feature_embedding_server(server_name);
        });
}
