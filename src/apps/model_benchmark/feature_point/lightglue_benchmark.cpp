/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: lightglue_benchmark.cpp
 * Date: 23-11-6
 ************************************************/

// lightglue benckmark tool

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "models/feature_point/lightglue.h"

using jinq::common::FilePathUtil;
using jinq::common::Timestamp;
using jinq::common::CvUtils;
using jinq::models::io_define::common_io::pair_mat_input;
using jinq::models::io_define::feature_point::matched_fp;
using jinq::models::io_define::feature_point::std_feature_point_match_output;
using jinq::models::feature_point::LightGlue;

int main(int argc, char** argv) {

    if (argc != 2 && argc != 4) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path [test_image_path]";
        return -1;
    }

    std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;

    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }

    std::string src_input_image_path;
    std::string dst_input_image_path;
    if (argc == 4) {
        src_input_image_path = argv[2];
        LOG(INFO) << "src input test image path: " << src_input_image_path;
        dst_input_image_path = argv[3];
        LOG(INFO) << "dst input test image path: " << dst_input_image_path;
    } else {
        src_input_image_path = "../demo_data/model_test_input/feature_point/match_test_01.jpg";
        LOG(INFO) << "use default src input test image path: " << src_input_image_path;
        dst_input_image_path = "../demo_data/model_test_input/feature_point/match_test_02.jpg";
        LOG(INFO) << "use default dst input test image path: " << dst_input_image_path;
    }

    if (!FilePathUtil::is_file_exist(src_input_image_path)) {
        LOG(INFO) << "test src input image file: " << src_input_image_path << " not exist";
        return -1;
    }
    if (!FilePathUtil::is_file_exist(dst_input_image_path)) {
        LOG(INFO) << "test dst input image file: " << dst_input_image_path << " not exist";
        return -1;
    }

    // construct model input
    cv::Mat src_input_image = cv::imread(src_input_image_path, cv::IMREAD_COLOR);
    cv::Mat dst_input_image = cv::imread(dst_input_image_path, cv::IMREAD_COLOR);
    struct pair_mat_input model_input{src_input_image, dst_input_image};

    // construct extractor
    auto matcher = std::make_unique<LightGlue<pair_mat_input, std_feature_point_match_output >>();
    auto cfg = toml::parse(cfg_file_path);
    matcher->init(cfg);
    if (!matcher->is_successfully_initialized()) {
        LOG(INFO) << "lightglue matcher init failed";
        return -1;
    }

    // run matcher
    std_feature_point_match_output model_output;
    int loop_times = 100;
    LOG(INFO) << "src input test image size: " << src_input_image.size();
    LOG(INFO) << "dst input test image size: " << dst_input_image.size();
    LOG(INFO) << "matcher run loop times: " << loop_times;
    LOG(INFO) << "start lightglue benchmark at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        matcher->run(model_input, model_output);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    // vis match result
    cv::Mat vis_result;
    CvUtils::visualize_fp_match_result(src_input_image, dst_input_image, model_output, vis_result);
    std::string output_file_name = FilePathUtil::get_file_name(src_input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_lightglue_result.png";
    std::string output_path = FilePathUtil::concat_path("../demo_data/model_test_input/feature_point", output_file_name);
    cv::imwrite(output_path, vis_result);
    LOG(INFO) << "feature point match result image has been written into: " << output_path;

    return 0;
}
