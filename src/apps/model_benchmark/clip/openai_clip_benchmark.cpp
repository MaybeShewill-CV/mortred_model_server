/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: openai_clip_benchmark.cpp
 * Date: 23-8-3
 ************************************************/

// openai clip model bench mark

#include <glog/logging.h>
#include <toml/toml.hpp>

#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/clip/openai_clip.h"

using jinq::common::FilePathUtil;
using jinq::common::Timestamp;
using jinq::models::clip::OpenAiClip;

int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_stop_logging_if_full_disk = true;

    if (argc != 2 && argc != 3) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path";
        return -1;
    }

    std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }
    auto cfg = toml::parse(cfg_file_path);

    std::string input_image_path = "../demo_data/model_test_input/clip/fox.jpg";
    LOG(INFO) << "input image file path: " << input_image_path;
    if (!FilePathUtil::is_file_exist(input_image_path)) {
        LOG(INFO) << "input image file: " << input_image_path << " not exist";
        return -1;
    }
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_COLOR);

    std::vector<std::string> input_texts = {"a photo of fox", "a photo of dog", "a photo of cat"};

    // construct clip model
    OpenAiClip clip;
    auto status = clip.init(cfg);
    if (!clip.is_successfully_initialized()) {
        LOG(ERROR) << "init clip model failed status: " << status;
        return -1;
    }

    // benchmark visual feature extraction
    int loop_times = 50;
    LOG(INFO) << "input test image size: " << input_image.size();
    LOG(INFO) << "visual feature extractor run loop times: " << loop_times;
    LOG(INFO) << "-- start clip vis feats extraction at: " << Timestamp::now().to_format_str();
    auto ts = Timestamp::now();
    std::vector<float> visual_feats;
    for (int i = 0; i < loop_times; ++i) {
        clip.get_visual_embedding(input_image, visual_feats);
    }
    auto cost_time = Timestamp::now() - ts;
    LOG(INFO) << "-- benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "-- cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    // benchmark textual feature extraction
    LOG(INFO) << "textual feature extractor run loop times: " << loop_times;
    LOG(INFO) << "-- start clip text feats extraction at: " << Timestamp::now().to_format_str();
    ts = Timestamp::now();
    std::vector<float> text_feats;
    for (int i = 0; i < loop_times; ++i) {
        clip.get_textual_embedding(input_texts[0], text_feats);
    }
    cost_time = Timestamp::now() - ts;
    LOG(INFO) << "-- benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "-- cost time: " << cost_time << "s, fps: " << loop_times / cost_time;

    // benchmark text to image similarity
    LOG(INFO) << "text to images run loop times: " << loop_times;
    LOG(INFO) << "-- start clip text to images similarity at: " << Timestamp::now().to_format_str();
    ts = Timestamp::now();
    std::vector<float> simi_scores;
    for (int i = 0; i < loop_times; ++i) {
        clip.texts2img(input_texts, input_image, simi_scores);
    }
    cost_time = Timestamp::now() - ts;
    LOG(INFO) << "-- benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "-- cost time: " << cost_time << "s, fps: " << loop_times / cost_time;
    LOG(INFO) << "-- text2imgs final scores: " << simi_scores[0] << ", " << simi_scores[1] << ", " << simi_scores[2];

    // benchmark image to text similarity
    LOG(INFO) << "images to text run loop times: " << loop_times;
    LOG(INFO) << "-- start clip images to text similarity at: " << Timestamp::now().to_format_str();
    ts = Timestamp::now();
    for (int i = 0; i < loop_times; ++i) {
        clip.imgs2text({input_image, input_image}, input_texts[0], simi_scores);
    }
    cost_time = Timestamp::now() - ts;
    LOG(INFO) << "-- benchmark ends at: " << Timestamp::now().to_format_str();
    LOG(INFO) << "-- cost time: " << cost_time << "s, fps: " << loop_times / cost_time;
    LOG(INFO) << "-- imgs2text final scores: " << simi_scores[0] << ", " << simi_scores[1];

    return 0;
}
