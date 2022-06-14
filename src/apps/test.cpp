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
#include "workflow/WFHttpServer.h"
#include "glog/logging.h"

#include "common/base64.h"
#include "common/status_code.h"
#include "common/time_stamp.h"

using morted::common::Base64;
using morted::common::StatusCode;
using morted::common::Timestamp;

void do_segmentation() {
    LOG(INFO) << "start doing segmentation: " << Timestamp::now().to_format_str();
    std::this_thread::sleep_for(std::chrono::milliseconds(10001));
}

void do_segmentation_callback(const WFGoTask* task) {
    auto state = task->get_state();
    if (state == WFT_STATE_ABORTED) {
        LOG(ERROR) << "segmentation task timeout at: " << Timestamp::now().to_format_str();
    } else if (state == WFT_STATE_SUCCESS) {
        LOG(INFO) << "segmentation task state success at: " << Timestamp::now().to_format_str();
    } else {
        LOG(INFO) << "segmentation task state: " << state;
    }
}

int main(int argc, char** argv) {

    WFFacilities::WaitGroup wait_group(1);
    auto* series = Workflow::create_series_work(
                       WFTaskFactory::create_empty_task(),
    [&](const SeriesWork * work) {
        LOG(INFO) << "Series complete at: " << Timestamp::now().to_format_str();
    });

    auto* segmentaiton_work = WFTaskFactory::create_timedgo_task(10, 0, "do_segmentation", do_segmentation);
    segmentaiton_work->set_callback(do_segmentation_callback);

    *series << segmentaiton_work;
    series->start();

    LOG(INFO) << "start whole process at: " << Timestamp::now().to_format_str();
    wait_group.wait();
    return 1;
}