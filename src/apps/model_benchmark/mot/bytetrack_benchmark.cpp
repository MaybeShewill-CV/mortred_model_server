/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: bytetrack_benchmark.cpp
 * Date: 24-5-31
 ************************************************/

// byte track benchmark tool

#include <glog/logging.h>
#include "toml/toml.hpp"
#include "indicators/indicators.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "models/mot/byte_tracker/byte_tracker.h"
#include "models/object_detection/yolov5_detector.h"

using jinq::common::FilePathUtil;
using jinq::common::Timestamp;
using jinq::common::CvUtils;
using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::object_detection::std_object_detection_output;
using jinq::models::object_detection::YoloV5Detector;
using jinq::models::mot::byte_tracker::ByteTracker;
using jinq::models::mot::byte_tracker::STrack;
using jinq::models::mot::byte_tracker::Object;

int main(int argc, char** argv) {

    if (argc != 2 && argc != 3 && argc != 4) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path [test_image_dir] [save_dir]";
        return -1;
    }

    std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }

    std::string input_image_dir = "../demo_data/model_test_input/mot";
    if (argc >= 3) {
        input_image_dir = argv[2];
    }

    std::string output_save_dir = "../demo_data/model_test_input/mot";
    if (argc >= 4) {
        output_save_dir = argv[3];
    }

    auto cfg = toml::parse(cfg_file_path);
    auto tracker = std::make_unique<ByteTracker>();
    tracker->init(cfg);
    if (!tracker->is_successfully_initialized()) {
        LOG(INFO) << "init tracker failed";
        return -1;
    }

    YoloV5Detector<mat_input, std_object_detection_output> detector;
    cfg = toml::parse("../conf/model/object_detection/yolov5/yolov5_config.ini");
    detector.init(cfg);

    std::vector<std::string> file_input_paths;
    cv::glob(input_image_dir + "/*.jpg", file_input_paths);
    std::sort(file_input_paths.begin(), file_input_paths.end(), [](const std::string& a, const std::string& b) -> bool {
       auto a_name = FilePathUtil::get_file_name(a);
       auto b_name = FilePathUtil::get_file_name(b);
       auto a_prefix = a_name.substr(0, a_name.find_first_of('.'));
       auto b_prefix = b_name.substr(0, b_name.find_first_of('.'));
       return std::stod(a_prefix) < std::stod(b_prefix);
    });

    // init progress bar
    auto progress_bar = std::make_unique<indicators::BlockProgressBar>();
    progress_bar->set_option(indicators::option::BarWidth{80});
    progress_bar->set_option(indicators::option::Start{"["});
    progress_bar->set_option(indicators::option::End{"]"});
    progress_bar->set_option(indicators::option::ForegroundColor{indicators::Color::white});
    progress_bar->set_option(indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
    progress_bar->set_option(indicators::option::ShowElapsedTime{true});
    progress_bar->set_option(indicators::option::ShowPercentage{true});
    progress_bar->set_option(indicators::option::ShowRemainingTime(true));

    int idx = 0;
    for (auto& file_path : file_input_paths) {
        cv::Mat input_image = cv::imread(file_path, cv::IMREAD_COLOR);
        mat_input det_in {input_image};
        std_object_detection_output det_out;
        detector.run(det_in, det_out);

        // track
        std::vector<STrack> output_stracks = tracker->update(det_out);
        for (auto & output_strack : output_stracks) {
            std::vector<float> tlwh = output_strack.tlwh;
            std::vector<int> tlwh_int;
            for (auto& value : tlwh) {
                tlwh_int.push_back(static_cast<int>(value));
            }
            if (tlwh[2] * tlwh[3] > 20) {
                cv::Scalar s = tracker->get_color(output_strack.track_id);
                cv::putText(input_image, cv::format("%d", output_strack.track_id), cv::Point(tlwh_int[0], tlwh_int[1] - 5),
                            0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(input_image, cv::Rect(tlwh_int[0], tlwh_int[1], tlwh_int[2], tlwh_int[3]), s, 2);
            }
        }
        std::string output_name = "track_output_" + std::to_string(idx) + ".jpg";
        std::string output_path = FilePathUtil::concat_path(output_save_dir, output_name);
        cv::imwrite(output_path, input_image);
        idx++;
        progress_bar->set_progress((static_cast<float>(idx) / static_cast<float>(file_input_paths.size())) * 100.0f);
    }
    progress_bar->mark_as_completed();

    return 1;
}
