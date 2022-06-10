/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: test.cpp
* Date: 22-6-2
************************************************/

#include <iostream>
#include <string>

#include "workflow/WFFacilities.h"
#include "workflow/WFTaskFactory.h"
#include "glog/logging.h"

#include "common/base64.h"
#include "common/status_code.h"
#include "common/time_stamp.h"

using morted::common::Base64;
using morted::common::StatusCode;
using morted::common::Timestamp;

void do_segmentation() {
    auto segmentation_timeout_timer = WFTaskFactory::create_timer_task(20, 0,[&](const WFTimerTask* task) {
        LOG(INFO) << "segmentation task timeout at:" << Timestamp::now().to_format_str();
        WFTaskFactory::count_by_name("count");
    });
    segmentation_timeout_timer->start();
    LOG(INFO) << "start doing segmentation: " << Timestamp::now().to_format_str();
    std::this_thread::sleep_for(std::chrono::seconds(25));
}

int main(int argc, char** argv) {

    WFHttpServer server([](WFHttpTask *task) {
        task->get_resp()->append_output_body("<html>Hello World!</html>");
    });

    if (server.start(80) == 0) { // start server on port 8888
        getchar(); // press "Enter" to end.
        server.stop();
    }

//    WFFacilities::WaitGroup wait_group(1);
//    auto* series = Workflow::create_series_work(
//            WFTaskFactory::create_empty_task(),
//            [&](const SeriesWork* work){
//                LOG(INFO) << "Series complete at: " << Timestamp::now().to_format_str();
//            });
//
//    WFTimerTask* get_image_data = WFTaskFactory::create_timer_task(
//            1000000 * 5, [&](const WFTimerTask* task) {
//                LOG(INFO) << "successfully get image data at: " << Timestamp::now().to_format_str();
//            });
//    WFCounterTask *counter = WFTaskFactory::create_counter_task("count", 1, [&](const WFCounterTask* task) {
//        LOG(INFO) << "counter task complete at: " << Timestamp::now().to_format_str();
//        auto* postprocess_work = WFTaskFactory::create_go_task("postprocess", [&]() {
//            LOG(INFO) << "do postprocess task at: " << Timestamp::now().to_format_str();
//        });
//        *series_of(task) << postprocess_work;
//    });
//    counter->start();
//
//    auto* segmentaiton_work = WFTaskFactory::create_go_task("do_segmentation", do_segmentation);
//    segmentaiton_work->set_callback([&](const WFGoTask* task) {
//        LOG(INFO) << "complete segmentation at:" << Timestamp::now().to_format_str();
//        WFTaskFactory::count_by_name("count");
//    });
//
//    *series << get_image_data;
//    *series << segmentaiton_work;
//    series->start();
//
//    LOG(INFO) << "start whole process at: " << Timestamp::now().to_format_str();
//    wait_group.wait();
    return 1;
}