/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* File: yolov5_detection_server.cpp
* Date: 2026-08-19
************************************************/

// yolov5_detection server tool

#include "apps/common/model_server_main.h"
#include "factory/obj_detection_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "YOLOV5_DETECTION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::object_detection::create_yolov5_det_server(server_name);
        });
}
