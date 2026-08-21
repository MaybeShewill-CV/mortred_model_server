/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: yolov7_detection_server.cpp
 * Date: 26-8-19
 ************************************************/

// yolov7_detection server tool

#include "apps/common/model_server_main.h"
#include "factory/obj_detection_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "YOLOV7_DETECTION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::object_detection::create_yolov7_det_server(server_name);
        });
}
