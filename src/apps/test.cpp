/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: test.cpp
* Date: 22-6-2
************************************************/

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFFacilities.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "glog/logging.h"

#include "common/base64.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "factory/classification_task.h"

using jinq::common::Base64;
using jinq::common::StatusCode;
using jinq::common::Timestamp;
using jinq::models::io_define::common_io::file_input;
using jinq::models::io_define::classification::std_classification_output;
using jinq::factory::classification::create_resnet_classifier;

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

void test_timedgo_task() {
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
}

void test_parallel_wget(int parallel_count = 200) {

    WFFacilities::WaitGroup wait_group(1);

    ParallelWork *pwork = Workflow::create_parallel_work([](const ParallelTask* task) {
        LOG(INFO) << "pwork end at: " << Timestamp::now().to_format_str();
    });
    SeriesWork *series;
    WFHttpTask *new_task;
    protocol::HttpRequest *req;
    int i;

    for (i = 0; i < parallel_count; i++) {
        std::string url = "http://localhost:8094/welcome";
        new_task = WFTaskFactory::create_http_task(url, 5, 5, [&i](WFHttpTask *task) {
            auto* resp = task->get_resp();
            std::string resp_body_str = protocol::HttpUtil::decode_chunked_body(resp);
            LOG(INFO) << "parallel worker: " << i << ", resp: " << resp_body_str;
        });
        req = new_task->get_req();
        series = Workflow::create_series_work(new_task, nullptr);
        pwork->add_series(series);
    }
    auto t_start = Timestamp::now();
    Workflow::start_series_work(pwork, [&wait_group, &t_start](const SeriesWork* work) {
        auto t_end = Timestamp::now();
        auto diff = t_end.micro_sec_since_epoch() - t_start.micro_sec_since_epoch();
        LOG(INFO) << "diff time: " << diff;
        wait_group.done();
    });
    wait_group.wait();
}

int main(int argc, char** argv) {

    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = -1;
    settings.handler_threads = 500;
    WORKFLOW_library_init(&settings);

    // test timed go task
    // test_timedgo_task();

    // test parallel wget
    // int parallel_nums = std::stoi(argv[1]);
    // test_parallel_wget(parallel_nums);

    auto resnet_1 = create_resnet_classifier<file_input, std_classification_output>("resnet_1");
    auto resnet_2 = create_resnet_classifier<file_input, std_classification_output>("resnet_1");

    LOG(INFO) << "resnet_1 address: " << resnet_1.get();
    LOG(INFO) << "resnet_2 address: " << resnet_2.get();

    return 1;
}