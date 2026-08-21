/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: cls_cond_ddim_sampler_benchmark.cpp
 * Date: 26-8-19
 ************************************************/

// cls-cond-ddim-sampler benchmark tool

#include "apps/common/benchmark_runner.h"
#include "models/diffusion/cls_cond_ddim_sampler.h"

using jinq::models::io_define::diffusion::std_cls_cond_ddim_input;
using jinq::models::io_define::diffusion::std_cls_cond_ddim_output;

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<std_cls_cond_ddim_input, std_cls_cond_ddim_output> spec;
    spec.model_name = "cls_cond_ddim";
    spec.display_name = "class cond ddim sampler";
    spec.usage =
        "exe config_file_path cls_id [sample_size(default: 128)] [sample_steps(default: 10)] "
        "[save_all_mid_results(default: true)]";
    spec.loops = 1;
    spec.warmup = false;
    spec.args_ok = [](int argc) { return argc >= 2 && argc <= 6; };
    spec.input_ok = [](const std_cls_cond_ddim_input&) { return true; };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        int cls_id = 0;
        if (argc >= 3) {
            cls_id = std::stoi(argv[2]);
        }
        int sample_size = 128;
        if (argc >= 4) {
            sample_size = std::stoi(argv[3]);
        }
        int sample_steps = 10;
        if (argc >= 5) {
            sample_steps = std::stoi(argv[4]);
        }
        bool save_all_mid_results = true;
        if (argc >= 6) {
            save_all_mid_results = std::stoi(argv[5]) == 1;
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
    spec.image_path_of = [](int, char**) {
        return std::string("../demo_data/model_test_input/diffusion/ddim");
    };
    spec.make_model = [](const std::string&) {
        return std::unique_ptr<
            jinq::models::BaseAiModel<std_cls_cond_ddim_input, std_cls_cond_ddim_output>>(
            new jinq::models::diffusion::ClsCondDDIMSampler<std_cls_cond_ddim_input,
                                                            std_cls_cond_ddim_output>());
    };
    spec.handle_output = [](const std_cls_cond_ddim_input& in,
                            const std_cls_cond_ddim_output& out, const std::string& save_dir) {
        if (in.save_all_mid_results) {
            auto stacked_sampled_image =
                jinq::common::CvUtils::stack_multiple_ddpm_images(out.sampled_images);
            auto stacked_predict_x0_image =
                jinq::common::CvUtils::stack_multiple_ddpm_images(out.predicted_x0);
            cv::imwrite(jinq::common::FilePathUtil::concat_path(
                            save_dir, "stacked_sampled_image.png"),
                        stacked_sampled_image);
            cv::imwrite(jinq::common::FilePathUtil::concat_path(
                            save_dir, "stacked_predict_x0_image.png"),
                        stacked_predict_x0_image);
        } else {
            cv::imwrite(jinq::common::FilePathUtil::concat_path(save_dir, "sample-step-0.png"),
                        out.sampled_images[0]);
            cv::imwrite(jinq::common::FilePathUtil::concat_path(
                            save_dir, "predict_x0-step-0.png"),
                        out.predicted_x0[0]);
        }
        LOG(INFO) << "sampled result image has been written into: " << save_dir;
    };
    return jinq::apps::run_benchmark(argc, argv, spec);
}
