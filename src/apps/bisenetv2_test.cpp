/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: bisenetv2_test.cpp
* Date: 22-6-7
************************************************/

#include <random>
#include <map>

#include <glog/logging.h>
#include <toml/toml.hpp>
#include <opencv2/opencv.hpp>

#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "factory/scene_segmentation_task.h"

using morted::common::Timestamp;
using morted::models::io_define::common_io::file_input;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::common_io::base64_input;
using morted::models::io_define::scene_segmentation::common_out;
using morted::factory::scene_segmentation::create_bisenetv2_segmentor;

std::map<int, cv::Scalar> generate_color_map(int class_counts) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255);

    std::set<int> color_set_r;
    std::set<int> color_set_g;
    std::set<int> color_set_b;
    std::map<int, cv::Scalar> color_map;
    int class_id = 0;

    while (color_map.size() != class_counts) {
        int r = distrib(gen);
        int g = distrib(gen);
        int b = distrib(gen);
        cv::Scalar color(b, g, r);

        if (color_set_r.find(r) != color_set_r.end() && color_set_g.find(g) != color_set_g.end()
            && color_set_b.find(b) != color_set_b.end()) {
            continue;
        } else {
            color_map.insert(std::make_pair(class_id, color));
            color_set_r.insert(r);
            color_set_g.insert(g);
            color_set_b.insert(b);
            class_id++;
        }
    }

    return color_map;
}

int main(int argc, char** argv) {

    if (argc != 2) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path";
        return -1;
    }

    LOG(INFO) << "cfg file path: " << argv[1];

    auto cfg = toml::parse(argv[1]);

    file_input file_in{};
    file_in.input_image_path = "../demo_data/model_test_input/scene_segmentation/cityscapes/test_01.png";
    mat_input mat_in{};
    base64_input base64_in{};
    std::vector<common_out> out;

    auto segmentor_1 = create_bisenetv2_segmentor<file_input, common_out>("bisenetv2_fc_worker1");
    segmentor_1->init(cfg);
    Timestamp ts = Timestamp::now();
    for (int i = 0; i < 50; ++i) {
        segmentor_1->run(file_in, out);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "bisenetv2 file in cost time: " << cost_time << "s";

    mat_in.input_image = cv::imread(
            "../demo_data/model_test_input/scene_segmentation/cityscapes/test_01.png", cv::IMREAD_UNCHANGED);
    auto segmentor_2 = create_bisenetv2_segmentor<mat_input, common_out>("bisenetv2_mc_worker1");
    segmentor_2->init(cfg);
    out.clear();

    ts = Timestamp::now();
    for (int i = 0; i < 50; ++i) {
        segmentor_2->run(mat_in, out);
    }
    cost_time = Timestamp::now() - ts;
    LOG(INFO) << "bisenetv2 mat in cost time: " << cost_time << "s";
    LOG(INFO) << "time stamp: " << ts.to_str();
    LOG(INFO) << "time stamp format str: " << ts.to_format_str();

    cv::Mat segmentation_result = out[0].segmentation_result;
    cv::Mat segmentation_color = cv::Mat::zeros(segmentation_result.size(), CV_8UC3);
    auto color_pool = generate_color_map(30);
    for (auto row= 0; row < segmentation_result.rows; ++row) {
        for (auto col = 0; col < segmentation_result.cols; ++col) {
            auto cls_id = segmentation_result.at<int>(row, col);
            cv::Scalar color(0, 0, 0);
            if (color_pool.find(cls_id) != color_pool.end()) {
                color = color_pool[cls_id];
            }
            segmentation_color.at<cv::Vec3b>(row, col)[0] = color[0];
            segmentation_color.at<cv::Vec3b>(row, col)[1] = color[1];
            segmentation_color.at<cv::Vec3b>(row, col)[2] = color[2];
        }
    }
    cv::imwrite(
            "../demo_data/model_test_input/scene_segmentation/cityscapes/segmentation_result.png",
            segmentation_color);

    return 1;
}