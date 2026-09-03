/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: custom_drivers.cpp
 * Date: 26-9-3
 ************************************************/

#include "apps/benchmark/custom_drivers.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <indicators/indicators.hpp>
#include <opencv2/opencv.hpp>
#include <toml/toml.hpp>

#include "apps/common/benchmark_runner.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "factory/clip_task.h"
#include "factory/feature_point_task.h"
#include "factory/sam_task.h"
#include "models/diffusion/cls_cond_ddim_sampler.h"
#include "models/diffusion/ddim_sampler.h"
#include "models/diffusion/ddpm_sampler.h"
#include "models/diffusion/ldm_sampler.h"
#include "models/model_io_define.h"
#include "models/mot/byte_tracker/byte_tracker.h"
#include "models/object_detection/yolov5_detector.h"

namespace jinq {
namespace apps {
namespace benchmark {
namespace {

using jinq::common::CvUtils;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

int run_ddpm_benchmark(int argc, char **argv) {
    using jinq::models::io_define::diffusion::std_ddpm_input;
    using jinq::models::io_define::diffusion::std_ddpm_output;
    BenchmarkSpec<std_ddpm_input, std_ddpm_output> spec;
    spec.model_name = "DDPM";
    spec.display_name = "ddpm sampler";
    spec.usage = "exe --model DDPM config_file_path [save_dir] [save_all_mid_results] [use_fixed_noise]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int n) { return n >= 2 && n <= 5; };
    spec.input_ok = [](const std_ddpm_input &) { return true; };
    spec.make_input = [](int n, char **a, const toml::table &) {
        bool save_all_mid_results = false;
        if (n >= 4) {
            save_all_mid_results = std::stoi(a[3]) == 1;
        }
        bool use_fixed_noise = false;
        if (n >= 5) {
            use_fixed_noise = std::stoi(a[4]) == 1;
        }
        std_ddpm_input in;
        in.sample_size = cv::Size(128, 128);
        in.timestep = 1000;
        in.channels = 3;
        in.save_all_mid_results = save_all_mid_results;
        in.use_fixed_noise_for_psample = use_fixed_noise;
        return in;
    };
    spec.image_path_of = [](int n, char **a) {
        return std::string(n >= 3 ? a[2] : "../demo_data/model_test_input/diffusion/ddpm");
    };
    spec.make_model = [](const std::string &) {
        return std::unique_ptr<jinq::models::BaseAiModel<std_ddpm_input, std_ddpm_output>>(
            new jinq::models::diffusion::DDPMSampler<std_ddpm_input, std_ddpm_output>());
    };
    spec.handle_output = [](const std_ddpm_input &in, const std_ddpm_output &out, const std::string &save_dir) {
        if (in.save_all_mid_results) {
            for (int idx = 0; idx < in.timestep; ++idx) {
                const std::string save_name = "sample-step-" + std::to_string(in.timestep - 1 - idx) + ".png";
                cv::imwrite(FilePathUtil::concat_path(save_dir, save_name), out.out_images[idx]);
            }
        } else {
            cv::imwrite(FilePathUtil::concat_path(save_dir, "sample-step-0.png"), out.out_images[0]);
        }
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return run_benchmark(argc, argv, spec);
}

int run_ddim_benchmark(int argc, char **argv) {
    using jinq::models::io_define::diffusion::std_ddim_input;
    using jinq::models::io_define::diffusion::std_ddim_output;
    BenchmarkSpec<std_ddim_input, std_ddim_output> spec;
    spec.model_name = "DDIM";
    spec.display_name = "ddim sampler";
    spec.usage = "exe --model DDIM config_file_path [sample_size] [sample_steps] [save_all_mid_results]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int n) { return n >= 2 && n <= 5; };
    spec.input_ok = [](const std_ddim_input &) { return true; };
    spec.make_input = [](int n, char **a, const toml::table &) {
        int sample_size = 256;
        if (n >= 3) {
            sample_size = std::stoi(a[2]);
        }
        int sample_steps = 10;
        if (n >= 4) {
            sample_steps = std::stoi(a[3]);
        }
        bool save_all_mid_results = true;
        if (n >= 5) {
            save_all_mid_results = std::stoi(a[4]) == 1;
        }
        std_ddim_input in;
        in.sample_size = cv::Size(sample_size, sample_size);
        in.total_steps = 1000;
        in.sample_steps = sample_steps;
        in.channels = 3;
        in.save_all_mid_results = save_all_mid_results;
        in.eta = 1.0f;
        return in;
    };
    spec.image_path_of = [](int, char **) { return std::string("../demo_data/model_test_input/diffusion/ddim"); };
    spec.make_model = [](const std::string &) {
        return std::unique_ptr<jinq::models::BaseAiModel<std_ddim_input, std_ddim_output>>(
            new jinq::models::diffusion::DDIMSampler<std_ddim_input, std_ddim_output>());
    };
    spec.handle_output = [](const std_ddim_input &in, const std_ddim_output &out, const std::string &save_dir) {
        if (in.save_all_mid_results) {
            cv::imwrite(FilePathUtil::concat_path(save_dir, "stacked_sampled_image.png"),
                        CvUtils::stack_multiple_ddpm_images(out.sampled_images));
            cv::imwrite(FilePathUtil::concat_path(save_dir, "stacked_predict_x0_image.png"),
                        CvUtils::stack_multiple_ddpm_images(out.predicted_x0));
        } else {
            cv::imwrite(FilePathUtil::concat_path(save_dir, "sample-step-0.png"), out.sampled_images[0]);
            cv::imwrite(FilePathUtil::concat_path(save_dir, "predict_x0-step-0.png"), out.predicted_x0[0]);
        }
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return run_benchmark(argc, argv, spec);
}

int run_cls_cond_ddim_benchmark(int argc, char **argv) {
    using jinq::models::io_define::diffusion::std_cls_cond_ddim_input;
    using jinq::models::io_define::diffusion::std_cls_cond_ddim_output;
    BenchmarkSpec<std_cls_cond_ddim_input, std_cls_cond_ddim_output> spec;
    spec.model_name = "CLS_COND_DDIM";
    spec.display_name = "class cond ddim sampler";
    spec.usage = "exe --model CLS_COND_DDIM config_file_path [cls_id] [sample_size] [sample_steps] [save_all_mid_results]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int n) { return n >= 2 && n <= 6; };
    spec.input_ok = [](const std_cls_cond_ddim_input &) { return true; };
    spec.make_input = [](int n, char **a, const toml::table &) {
        int cls_id = 0;
        if (n >= 3) {
            cls_id = std::stoi(a[2]);
        }
        int sample_size = 128;
        if (n >= 4) {
            sample_size = std::stoi(a[3]);
        }
        int sample_steps = 10;
        if (n >= 5) {
            sample_steps = std::stoi(a[4]);
        }
        bool save_all_mid_results = true;
        if (n >= 6) {
            save_all_mid_results = std::stoi(a[5]) == 1;
        }
        std_cls_cond_ddim_input in;
        in.sample_size = cv::Size(sample_size, sample_size);
        in.total_steps = 1000;
        in.sample_steps = sample_steps;
        in.channels = 3;
        in.save_all_mid_results = save_all_mid_results;
        in.eta = 1.0f;
        in.cls_id = cls_id;
        return in;
    };
    spec.image_path_of = [](int, char **) { return std::string("../demo_data/model_test_input/diffusion/ddim"); };
    spec.make_model = [](const std::string &) {
        return std::unique_ptr<jinq::models::BaseAiModel<std_cls_cond_ddim_input, std_cls_cond_ddim_output>>(
            new jinq::models::diffusion::ClsCondDDIMSampler<std_cls_cond_ddim_input, std_cls_cond_ddim_output>());
    };
    spec.handle_output = [](const std_cls_cond_ddim_input &in, const std_cls_cond_ddim_output &out,
                            const std::string &save_dir) {
        if (in.save_all_mid_results) {
            cv::imwrite(FilePathUtil::concat_path(save_dir, "stacked_sampled_image.png"),
                        CvUtils::stack_multiple_ddpm_images(out.sampled_images));
            cv::imwrite(FilePathUtil::concat_path(save_dir, "stacked_predict_x0_image.png"),
                        CvUtils::stack_multiple_ddpm_images(out.predicted_x0));
        } else {
            cv::imwrite(FilePathUtil::concat_path(save_dir, "sample-step-0.png"), out.sampled_images[0]);
            cv::imwrite(FilePathUtil::concat_path(save_dir, "predict_x0-step-0.png"), out.predicted_x0[0]);
        }
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return run_benchmark(argc, argv, spec);
}

int run_ldm_benchmark(int argc, char **argv) {
    using jinq::models::io_define::diffusion::DDPMSampler_Type;
    using jinq::models::io_define::diffusion::std_ldm_input;
    using jinq::models::io_define::diffusion::std_ldm_output;
    BenchmarkSpec<std_ldm_input, std_ldm_output> spec;
    spec.model_name = "LDM";
    spec.display_name = "ldm sampler";
    spec.usage = "exe --model LDM config_file_path [sample_size] [sample_steps] [sampler_type]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int n) { return n >= 2 && n <= 5; };
    spec.input_ok = [](const std_ldm_input &) { return true; };
    spec.make_input = [](int n, char **a, const toml::table &) {
        int sample_size = 256;
        if (n >= 3) {
            sample_size = std::stoi(a[2]);
        }
        int sample_steps = 100;
        if (n >= 4) {
            sample_steps = std::stoi(a[3]);
        }
        auto sampler_type = DDPMSampler_Type::DDIM;
        if (n >= 5) {
            sampler_type = static_cast<DDPMSampler_Type>(std::stoi(a[4]));
        }
        std_ldm_input in;
        in.sample_size = cv::Size(sample_size, sample_size);
        in.step_size = sample_steps;
        if (sampler_type == DDPMSampler_Type::DDPM) {
            in.step_size = 1000;
        }
        in.downscale = 8;
        in.latent_dims = 4;
        in.latent_scale = 0.18215f;
        in.sampler_type = sampler_type;
        return in;
    };
    spec.image_path_of = [](int, char **) { return std::string("../demo_data/model_test_input/diffusion/ldm"); };
    spec.make_model = [](const std::string &) {
        return std::unique_ptr<jinq::models::BaseAiModel<std_ldm_input, std_ldm_output>>(
            new jinq::models::diffusion::LDMSampler<std_ldm_input, std_ldm_output>());
    };
    spec.handle_output = [](const std_ldm_input &, const std_ldm_output &out, const std::string &save_dir) {
        cv::imwrite(FilePathUtil::concat_path(save_dir, "ldm_sample_output.png"), out.sampled_image);
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return run_benchmark(argc, argv, spec);
}

} // namespace

int run_openai_clip_benchmark(int argc, char **argv) {
    using jinq::models::io_define::clip::clip_input;
    using jinq::models::io_define::clip::clip_output;
    using jinq::models::io_define::clip::ClipTaskType;

    if (argc != 2 && argc != 3) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe --model OPENAI_CLIP config_file_path [test_image_path]";
        return -1;
    }
    const std::string cfg_file_path = argv[1];
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }
    auto cfg_parsed = toml::parse_file(cfg_file_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(cfg_parsed.error().description());
        return -1;
    }
    auto cfg = std::move(cfg_parsed).table();

    std::string input_image_path = "../demo_data/model_test_input/clip/fox.jpg";
    if (argc == 3) {
        input_image_path = argv[2];
    }
    if (!FilePathUtil::is_file_exist(input_image_path)) {
        LOG(INFO) << "input image file: " << input_image_path << " not exist";
        return -1;
    }
    const cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    const std::vector<std::string> input_texts = {"a photo of fox", "a photo of dog", "a photo of cat"};

    auto clip = jinq::factory::clip::create_model("OPENAI_CLIP");
    if (clip == nullptr) {
        return -1;
    }
    clip->init(cfg);
    if (!clip->is_successfully_initialized()) {
        LOG(ERROR) << "init clip model failed";
        return -1;
    }

    clip_input model_input;
    clip_output model_output;
    constexpr int loop_times = 50;

    LOG(INFO) << "visual feature extractor run loop times: " << loop_times;
    auto ts = Timestamp::now();
    model_input = clip_input{};
    model_input.task_type = ClipTaskType::IMAGE_EMBEDDING;
    model_input.image = input_image;
    for (int i = 0; i < loop_times; ++i) {
        clip->run(model_input, model_output);
    }
    LOG(INFO) << "-- vis feats cost: " << Timestamp::now() - ts << "s";

    ts = Timestamp::now();
    model_input = clip_input{};
    model_input.task_type = ClipTaskType::TEXT_EMBEDDING;
    model_input.text = input_texts[0];
    for (int i = 0; i < loop_times; ++i) {
        clip->run(model_input, model_output);
    }
    LOG(INFO) << "-- text feats cost: " << Timestamp::now() - ts << "s";

    ts = Timestamp::now();
    model_input = clip_input{};
    model_input.task_type = ClipTaskType::TEXTS_TO_IMAGE;
    model_input.texts = input_texts;
    model_input.image = input_image;
    for (int i = 0; i < loop_times; ++i) {
        clip->run(model_input, model_output);
    }
    LOG(INFO) << "-- text2imgs cost: " << Timestamp::now() - ts << "s, scores: " << model_output.simi_scores[0]
              << ", " << model_output.simi_scores[1] << ", " << model_output.simi_scores[2];

    ts = Timestamp::now();
    model_input = clip_input{};
    model_input.task_type = ClipTaskType::IMAGES_TO_TEXT;
    model_input.images = {input_image, input_image};
    model_input.text = input_texts[0];
    for (int i = 0; i < loop_times; ++i) {
        clip->run(model_input, model_output);
    }
    LOG(INFO) << "-- imgs2text cost: " << Timestamp::now() - ts << "s, scores: " << model_output.simi_scores[0]
              << ", " << model_output.simi_scores[1];
    return 0;
}

int run_sam_predictor_benchmark(int argc, char **argv) {
    using jinq::models::io_define::segment_anything::sam_prompt_input;
    using jinq::models::io_define::segment_anything::std_sam_prompt_output;

    if (argc < 2) {
        LOG(INFO) << "usage: exe --model SAM_PREDICTOR config_file [image]";
        return -1;
    }
    const std::string config_file_path = argv[1];
    if (!FilePathUtil::is_file_exist(config_file_path)) {
        LOG(ERROR) << "config file path: " << config_file_path << " not exists";
        return -1;
    }
    auto sam_model = jinq::factory::segment_anything::create_sam_predictor<sam_prompt_input, std_sam_prompt_output>(
        "sam_predictor");
    auto cfg_parsed = toml::parse_file(config_file_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(cfg_parsed.error().description());
        return -1;
    }
    auto cfg = std::move(cfg_parsed).table();
    sam_model->init(cfg);
    if (!sam_model->is_successfully_initialized()) {
        LOG(ERROR) << "init sam failed";
        return -1;
    }

    std::string input_image_path = "../demo_data/model_test_input/sam/truck.jpg";
    if (argc >= 3) {
        input_image_path = argv[2];
    }
    if (!FilePathUtil::is_file_exist(input_image_path)) {
        LOG(ERROR) << "input image file path: " << input_image_path << " not exists";
        return -1;
    }
    const cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);
    std_sam_prompt_output masks;
    sam_prompt_input model_input;
    model_input.image = input_image;
    model_input.bboxes = {cv::Rect(483, 683, 158, 132), cv::Rect(220, 327, 430, 122), cv::Rect(77, 78, 58, 176),
                          cv::Rect(972, 464, 111, 52)};
    for (int idx = 0; idx < 10; ++idx) {
        const auto t_start = std::chrono::system_clock::now();
        sam_model->run(model_input, masks);
        const auto t_end = std::chrono::system_clock::now();
        const auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << " .... iter: " << idx + 1 << " encoding cost time: " << t_cost << " ms";
    }
    model_input.bboxes.clear();
    model_input.prompt_points = {{cv::Point2f(1524, 675)}, {cv::Point2f(1094, 381)}, {cv::Point2f(183, 587)}};
    sam_model->run(model_input, masks);
    std::string output_file_name = FilePathUtil::get_file_name(input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_sam_output.png";
    const std::string output_path =
        FilePathUtil::concat_path("../demo_data/model_test_input/sam", output_file_name);
    cv::Mat color_output;
    CvUtils::visualize_sam_output_masks(input_image, masks, color_output);
    cv::imwrite(output_path, color_output);
    LOG(INFO) << "sam prediction result image has been written into: " << output_path;
    return 0;
}

int run_fast_sam_benchmark(int argc, char **argv) {
    using jinq::models::io_define::common_io::mat_input;
    using jinq::models::io_define::segment_anything::std_fast_sam_output;

    if (argc < 2) {
        LOG(INFO) << "usage: exe --model FAST_SAM config_file [image]";
        return -1;
    }
    const std::string config_file_path = argv[1];
    if (!FilePathUtil::is_file_exist(config_file_path)) {
        LOG(ERROR) << "config file path: " << config_file_path << " not exists";
        return -1;
    }
    auto fast_sam_model =
        jinq::factory::segment_anything::create_fast_sam_segmentor<mat_input, std_fast_sam_output>("fast_sam");
    auto cfg_parsed = toml::parse_file(config_file_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(cfg_parsed.error().description());
        return -1;
    }
    auto cfg = std::move(cfg_parsed).table();
    fast_sam_model->init(cfg);
    if (!fast_sam_model->is_successfully_initialized()) {
        LOG(ERROR) << "init fast-sam failed";
        return -1;
    }

    std::string input_image_path = "../demo_data/model_test_input/sam/truck.jpg";
    if (argc >= 3) {
        input_image_path = argv[2];
    }
    const cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_UNCHANGED);
    std_fast_sam_output everything_mask;
    mat_input model_input{input_image};
    for (int i = 0; i < 10; ++i) {
        const auto t_start = std::chrono::system_clock::now();
        fast_sam_model->run(model_input, everything_mask);
        const auto t_end = std::chrono::system_clock::now();
        const auto t_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        LOG(INFO) << "... infer: " << i << ", cost time: " << t_cost;
    }
    cv::Mat everything_color_mask;
    CvUtils::colorize_sam_everything_mask(everything_mask, everything_color_mask);
    cv::Mat merge_result;
    cv::addWeighted(input_image, 0.65, everything_color_mask, 0.35, 0.0, merge_result);
    std::string output_file_name = FilePathUtil::get_file_name(input_image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_fastsam_everything_result.png";
    const std::string output_path =
        FilePathUtil::concat_path("../demo_data/model_test_input/sam", output_file_name);
    cv::imwrite(output_path, merge_result);
    LOG(INFO) << "fast-sam everything result image has been written into: " << output_path;
    return 0;
}

int run_lightglue_benchmark(int argc, char **argv) {
    using jinq::models::io_define::common_io::pair_mat_input;
    using jinq::models::io_define::feature_point::std_feature_point_match_output;

    BenchmarkSpec<pair_mat_input, std_feature_point_match_output> spec;
    spec.model_name = "LIGHTGLUE";
    spec.display_name = "lightglue matcher";
    spec.usage = "exe --model LIGHTGLUE config_file_path [src_image_path dst_image_path]";
    spec.loops = 100;
    spec.args_ok = [](int n) { return n == 2 || n == 4; };
    spec.input_ok = [](const pair_mat_input &in) {
        return !in.src_input_image.empty() && !in.dst_input_image.empty();
    };
    spec.make_input = [](int n, char **a, const toml::table &) {
        const char *src = "../demo_data/model_test_input/feature_point/match_test_01.jpg";
        const char *dst = "../demo_data/model_test_input/feature_point/match_test_02.jpg";
        if (n == 4) {
            src = a[2];
            dst = a[3];
        }
        pair_mat_input in;
        in.src_input_image = cv::imread(src, cv::IMREAD_COLOR);
        in.dst_input_image = cv::imread(dst, cv::IMREAD_COLOR);
        return in;
    };
    spec.image_path_of = [](int n, char **a) {
        return std::string(n == 4 ? a[2] : "../demo_data/model_test_input/feature_point/match_test_01.jpg");
    };
    spec.make_model = [](const std::string &) {
        return jinq::factory::feature_point::create_lightglue_matcher<pair_mat_input, std_feature_point_match_output>(
            "lightglue");
    };
    spec.handle_output = [](const pair_mat_input &in, const std_feature_point_match_output &out,
                            const std::string &image_path) {
        cv::Mat vis_result;
        CvUtils::visualize_fp_match_result(in.src_input_image, in.dst_input_image, out, vis_result);
        std::string output_file_name = FilePathUtil::get_file_name(image_path);
        output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) + "_LIGHTGLUE_result.png";
        const std::string output_path =
            FilePathUtil::concat_path("../demo_data/model_test_input/feature_point", output_file_name);
        cv::imwrite(output_path, vis_result);
        LOG(INFO) << "feature point match result image has been written into: " << output_path;
    };
    return run_benchmark(argc, argv, spec);
}

int run_byte_track_benchmark(int argc, char **argv) {
    using jinq::models::io_define::common_io::mat_input;
    using jinq::models::io_define::object_detection::std_object_detection_output;
    using jinq::models::mot::byte_tracker::ByteTracker;
    using jinq::models::object_detection::YoloV5Detector;

    if (argc != 2 && argc != 3 && argc != 4) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe --model BYTE_TRACK config_file_path [test_image_dir] [save_dir]";
        return -1;
    }
    const std::string cfg_file_path = argv[1];
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
    auto cfg_parsed = toml::parse_file(cfg_file_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(cfg_parsed.error().description());
        return -1;
    }
    auto cfg = std::move(cfg_parsed).table();
    auto tracker = std::make_unique<ByteTracker>();
    tracker->init(cfg);
    if (!tracker->is_successfully_initialized()) {
        LOG(INFO) << "init tracker failed";
        return -1;
    }

    YoloV5Detector<mat_input, std_object_detection_output> detector;
    cfg_parsed = toml::parse_file("../conf/model/object_detection/yolov5/yolov5_config.toml");
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(cfg_parsed.error().description());
        return -1;
    }
    cfg = std::move(cfg_parsed).table();
    detector.init(cfg);
    if (!detector.is_successfully_initialized()) {
        LOG(INFO) << "init yolov5 detector failed";
        return -1;
    }

    std::vector<std::string> file_input_paths;
    cv::glob(input_image_dir + "/*.jpg", file_input_paths);
    std::sort(file_input_paths.begin(), file_input_paths.end(), [](const std::string &a, const std::string &b) {
        auto a_name = FilePathUtil::get_file_name(a);
        auto b_name = FilePathUtil::get_file_name(b);
        auto a_prefix = a_name.substr(0, a_name.find_first_of('.'));
        auto b_prefix = b_name.substr(0, b_name.find_first_of('.'));
        return std::stod(a_prefix) < std::stod(b_prefix);
    });

    auto progress_bar = std::make_unique<indicators::BlockProgressBar>();
    progress_bar->set_option(indicators::option::BarWidth{80});
    progress_bar->set_option(indicators::option::Start{"["});
    progress_bar->set_option(indicators::option::End{"]"});
    progress_bar->set_option(indicators::option::ForegroundColor{indicators::Color::white});
    progress_bar->set_option(indicators::option::FontStyles{
        std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
    progress_bar->set_option(indicators::option::ShowElapsedTime{true});
    progress_bar->set_option(indicators::option::ShowPercentage{true});
    progress_bar->set_option(indicators::option::ShowRemainingTime(true));

    int idx = 0;
    for (auto &file_path : file_input_paths) {
        cv::Mat input_image = cv::imread(file_path, cv::IMREAD_COLOR);
        mat_input det_in{input_image};
        std_object_detection_output det_out;
        detector.run(det_in, det_out);
        const auto output_stracks = tracker->update(det_out);
        for (const auto &output_strack : output_stracks) {
            std::vector<float> tlwh = output_strack.tlwh;
            std::vector<int> tlwh_int;
            for (auto &value : tlwh) {
                tlwh_int.push_back(static_cast<int>(value));
            }
            if (tlwh[2] * tlwh[3] > 20) {
                const cv::Scalar s = tracker->get_color(output_strack.track_id);
                cv::putText(input_image, cv::format("%d", output_strack.track_id),
                            cv::Point(tlwh_int[0], tlwh_int[1] - 5), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(input_image, cv::Rect(tlwh_int[0], tlwh_int[1], tlwh_int[2], tlwh_int[3]), s, 2);
            }
        }
        const std::string output_name = "track_output_" + std::to_string(idx) + ".jpg";
        cv::imwrite(FilePathUtil::concat_path(output_save_dir, output_name), input_image);
        ++idx;
        progress_bar->set_progress((static_cast<float>(idx) / static_cast<float>(file_input_paths.size())) * 100.0f);
    }
    progress_bar->mark_as_completed();
    return 0;
}

int run_diffusion_family_benchmark(const std::string &model_section, int argc, char **argv) {
    if (model_section == "DDPM") {
        return run_ddpm_benchmark(argc, argv);
    }
    if (model_section == "DDIM") {
        return run_ddim_benchmark(argc, argv);
    }
    if (model_section == "CLS_COND_DDIM") {
        return run_cls_cond_ddim_benchmark(argc, argv);
    }
    if (model_section == "LDM") {
        return run_ldm_benchmark(argc, argv);
    }
    LOG(ERROR) << "no diffusion benchmark driver for " << model_section;
    return -1;
}

} // namespace benchmark
} // namespace apps
} // namespace jinq
