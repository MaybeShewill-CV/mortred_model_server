/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: dbnet_text_region_detect_server.cpp
 * Date: 26-8-19
 ************************************************/

// dbnet_text_region_detect server tool

#include "apps/common/model_server_main.h"
#include "factory/ocr_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "DBNET_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::ocr::create_dbtext_detection_server(server_name);
        });
}
