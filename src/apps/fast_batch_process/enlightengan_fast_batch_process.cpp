/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: enlightengan_fast_batch_process.cpp
 * Date: 22-6-17
 ************************************************/

// enlightengan fast batch process tool

#include <chrono>
#include <random>


#include "stl_container/concurrentqueue.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <toml/toml.hpp>
#include <workflow/WFFacilities.h>
#include <workflow/WFTaskFactory.h>


#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "factory/enhancement_task.h"
#include "models/enhancement/enlightengan.h"
#include "models/model_io_define.h"


using morted::common::CvUtils;
using morted::common::FilePathUtil;
using morted::common::StatusCode;
using morted::common::Timestamp;
using morted::factory::enhancement::create_enlightengan_enhancementor;
using morted::models::BaseAiModel;
using morted::models::enhancement::EnlightenGan;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::enhancement::std_enhancement_output;

#define TIME_COST(start, end) std::chrono::duration_cast<std::chrono::milliseconds>((end) - (start)).count()

namespace common {

std::vector<std::string> get_test_image_paths(const std::string &input_dir) {
    std::vector<std::string> image_paths;
    cv::glob(input_dir, image_paths, false);
    return image_paths;
}

static std::unique_ptr<EnlightenGan<mat_input, std_enhancement_output>> &get_enhancementor() {

    static std::unique_ptr<EnlightenGan<mat_input, std_enhancement_output> > enhancementor_ptr = nullptr;
    std::string cfg_file_path = "../weights/enhancement/low_light/config.ini";
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return enhancementor_ptr;
    }
    auto cfg = toml::parse(cfg_file_path);
    enhancementor_ptr.reset(new EnlightenGan<mat_input, std_enhancement_output>());
    auto status = enhancementor_ptr->init(cfg);
    if (status != StatusCode::OK) {
        LOG(INFO) << "init enhancementor failed";
        enhancementor_ptr.reset(nullptr);
        return enhancementor_ptr;
    }
    return enhancementor_ptr;
}
} // namespace common

namespace all_parallel_impl {

struct ProcessImgRet {
    cv::Mat input_image;
    cv::Mat output_image;
    std::string save_path;
};

using ReadTaskInput = moodycamel::ConcurrentQueue<std::string>;
using ReadTaskOutput = moodycamel::ConcurrentQueue<std::pair<std::string, cv::Mat>>;
using ProcessImgTaskInput = ReadTaskOutput;
using ProcessImgTaskOutput = moodycamel::ConcurrentQueue<ProcessImgRet>;
using SaveImgTaskInput = ProcessImgTaskOutput;

/***
 *
 * @param tracker
 * @param input_image
 */
void enhance_frame(cv::Mat &input_image, const std::string &vis_save_path, ProcessImgTaskOutput &process_task_output) {

    mat_input model_input{input_image};
    std_enhancement_output model_output;
    auto &enhancementor = common::get_enhancementor();
    enhancementor->run(model_input, model_output);

    ProcessImgRet ret;
    ret.input_image = input_image;
    ret.save_path = vis_save_path;
    model_output.enhancement_result.copyTo(ret.output_image);
    while (!process_task_output.try_enqueue(ret)) {
    }
}

/***
 *
 * @param ret
 */
void save_frame(const ProcessImgRet &ret) { cv::imwrite(ret.save_path, ret.output_image); }

/***
 *
 * @param file_path_queue
 * @param image_queue
 */
void read_task(ReadTaskInput &read_in, ReadTaskOutput &read_out) {
    while (true) {
        std::string file_path;

        if (read_in.try_dequeue(file_path)) {
            auto image = cv::imread(file_path, cv::IMREAD_UNCHANGED);
            auto image_name = FilePathUtil::get_file_name(file_path);

            // try enqueue element util success
            while (!read_out.try_enqueue(std::make_pair(image_name, image))) {
            }
        } // break if file path queue empty
        else {
            break;
        }
    }
}

/***
 *
 * @param image_queue
 * @param tracker
 * @param no_more_data_will_be_produced
 */
void process_task(ProcessImgTaskInput &process_task_input, ProcessImgTaskOutput &process_task_output, const std::string &vis_save_dir,
                  bool &no_more_data_will_be_produced) {
    while (true) {
        std::pair<std::string, cv::Mat> image_info;

        if (no_more_data_will_be_produced) {
            // if not more data will be produced and image queue is empty break
            if (!process_task_input.try_dequeue(image_info)) {
                break;
            } else {
                auto file_name = FilePathUtil::get_file_name(image_info.first);
                auto vis_save_path = FilePathUtil::concat_path(vis_save_dir, file_name);
                enhance_frame(image_info.second, vis_save_path, process_task_output);
            }
        } else {
            if (process_task_input.try_dequeue(image_info)) {
                auto file_name = FilePathUtil::get_file_name(image_info.first);
                auto vis_save_path = FilePathUtil::concat_path(vis_save_dir, file_name);
                enhance_frame(image_info.second, vis_save_path, process_task_output);
            }
        }
    }
}

/***
 *
 * @param save_task_input
 */
void save_task(SaveImgTaskInput &save_task_input, bool &no_more_data_will_be_produced) {
    while (true) {
        ProcessImgRet process_ret;

        if (no_more_data_will_be_produced) {
            // if not more data will be produced and image queue is empty break
            if (!save_task_input.try_dequeue(process_ret)) {
                break;
            } else {
                save_frame(process_ret);
            }
        } else {
            if (save_task_input.try_dequeue(process_ret)) {
                save_frame(process_ret);
            }
        }
    }
}

/***
 *
 * @param cfg_path
 * @param input_dir
 * @param save_dir
 */
void parallel_enhance_lowlight(const std::string &input_dir, const std::string &save_dir) {
    std::vector<std::string> input_image_paths = common::get_test_image_paths(input_dir);
    WFFacilities::WaitGroup wait_group(1);

    // read image file path
    ReadTaskInput read_task_input(input_image_paths.size());

    for (const auto &f_path : input_image_paths) {
        read_task_input.enqueue(f_path);
    }

    // init total parallel work
    const int image_queue_buffer_size = 500;
    ReadTaskOutput read_task_output(image_queue_buffer_size);
    bool no_more_data_will_be_readed = false;
    ProcessImgTaskOutput process_task_output(image_queue_buffer_size);
    bool no_more_data_will_be_processed = false;
    auto *pwork = Workflow::create_parallel_work([&](const ParallelWork *work) {
        LOG(INFO) << "pwork complete";
        LOG(INFO) << "file path queue size: " << read_task_input.size_approx();
        LOG(INFO) << "image queue size: " << read_task_output.size_approx();
        wait_group.done();
    });

    // add image reader to p_work
    auto *p_read_work = Workflow::create_parallel_work([&](const ParallelWork *work) {
        LOG(INFO) << "p_produce_work complete and all producers exit";
        no_more_data_will_be_readed = true;
    });
    const int producer_count = 2;
    for (int index = 0; index < producer_count; ++index) {
        auto *producer = WFTaskFactory::create_go_task("producer", read_task, std::ref(read_task_input), std::ref(read_task_output));
        auto *produce_series = Workflow::create_series_work(producer, nullptr);
        p_read_work->add_series(produce_series);
    }

    // add process img to p_work
    auto *p_processimg_work = Workflow::create_parallel_work([&](const ParallelWork *work) {
        LOG(INFO) << "p_process_img_work complete and all process img workers exit";
        no_more_data_will_be_processed = true;
    });
    auto processimg_task =
        WFTaskFactory::create_go_task("process img worker", process_task, std::ref(read_task_output), std::ref(process_task_output),
                                      std::ref(save_dir), std::ref(no_more_data_will_be_readed));
    auto *consume_series = Workflow::create_series_work(processimg_task, nullptr);
    p_processimg_work->add_series(consume_series);

    // add save img to p_work
    auto *p_saveimg_work = Workflow::create_parallel_work(
        [&](const ParallelWork *work) { LOG(INFO) << "p_save_img_work complete and all save img workers exit"; });
    const int save_img_worker_count = 2;
    for (int index = 0; index < save_img_worker_count; ++index) {
        auto *saver = WFTaskFactory::create_go_task("save img worker", save_task, std::ref(process_task_output),
                                                    std::ref(no_more_data_will_be_processed));
        auto *save_series = Workflow::create_series_work(saver, nullptr);
        p_saveimg_work->add_series(save_series);
    }

    // start total work
    pwork->add_series(Workflow::create_series_work(p_read_work, nullptr));
    pwork->add_series(Workflow::create_series_work(p_processimg_work, nullptr));
    pwork->add_series(Workflow::create_series_work(p_saveimg_work, nullptr));
    pwork->start();
    auto t_start = std::chrono::system_clock::now();
    wait_group.wait();
    LOG(INFO) << "all parallel cost time: " << TIME_COST(t_start, std::chrono::system_clock::now()) / 1000.0 << " seconds";
}
} // namespace all_parallel_impl

int main(int argc, char **argv) {

    if (argc != 4) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path input_image_dir save_dir";
        return -1;
    }

    std::string cfg_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << cfg_file_path;
    std::string input_image_dir = argv[2];
    LOG(INFO) << "input image dir path: " << input_image_dir;
    std::string save_image_dir = argv[3];
    LOG(INFO) << "save image dir path: " << save_image_dir;

    // init workflow global settings
    struct WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = 8;
    WORKFLOW_library_init(&settings);

    // total parallel test
    if (!FilePathUtil::is_dir_exist(save_image_dir)) {
        LOG(INFO) << "save dir: " << save_image_dir << " not exist";
        return -1;
    }
    all_parallel_impl::parallel_enhance_lowlight(input_image_dir, save_image_dir);
    return 1;
}
