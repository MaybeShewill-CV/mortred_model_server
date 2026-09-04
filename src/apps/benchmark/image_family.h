/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: image_family.h
 * Date: 26-9-3
 ************************************************/

#ifndef MORTRED_APPS_BENCHMARK_IMAGE_FAMILY_H
#define MORTRED_APPS_BENCHMARK_IMAGE_FAMILY_H

#include <fstream>
#include <functional>
#include <iterator>
#include <string>

#include "apps/common/benchmark_runner.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "server/generic_cv_server.h"

namespace jinq {
namespace apps {
namespace benchmark {

using ImageInput = jinq::server::ImageInput;

template <typename OUTPUT>
struct ImageFamilyHooks {
    std::string default_image;
    std::string output_dir;
    int loops = 100;
    bool warmup = true;
    std::function<void(const cv::Mat &src, const OUTPUT &out, const std::string &image_path,
                       const std::string &model_id)>
        handle_output;
};

inline ImageInput make_raw_image_input(const std::string &path) {
    ImageInput in;
    in.image.origin = jinq::models::io_define::common_io::byte_source::origin_kind::raw_bytes;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return in;
    }
    in.image.data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return in;
}

inline std::string result_image_path(const std::string &output_dir, const std::string &image_path,
                                     const std::string &model_id) {
    std::string name = jinq::common::FilePathUtil::get_file_name(image_path);
    const auto dot = name.find_last_of('.');
    if (dot != std::string::npos) {
        name = name.substr(0, dot);
    }
    return jinq::common::FilePathUtil::concat_path(output_dir, name + "_" + model_id + "_result.png");
}

template <typename OUTPUT>
int run_image_family_benchmark(
    const std::string &model_id, const std::string &display_name,
    const std::function<std::unique_ptr<jinq::models::BaseAiModel<ImageInput, OUTPUT>>()> &make_model,
    const ImageFamilyHooks<OUTPUT> &hooks, int argc, char **argv) {
    BenchmarkSpec<ImageInput, OUTPUT> spec;
    spec.model_name = model_id;
    spec.display_name = display_name;
    spec.usage = "exe --model " + model_id + " config_file_path [test_image_path]";
    spec.loops = hooks.loops;
    spec.warmup = hooks.warmup;
    spec.args_ok = standard_args_ok;
    spec.input_ok = [](const ImageInput &in) { return !in.image.data.empty(); };
    spec.make_input = [hooks](int argc, char **argv, const toml::table &) {
        const std::string path = standard_image_path(argc, argv, hooks.default_image);
        if (!jinq::common::FilePathUtil::is_file_exist(path)) {
            LOG(INFO) << "test input image file: " << path << " not exist";
            return ImageInput{};
        }
        ImageInput in = make_raw_image_input(path);
        if (in.image.data.empty()) {
            LOG(ERROR) << "failed to read image bytes: " << path;
        }
        return in;
    };
    spec.image_path_of = [hooks](int argc, char **argv) {
        return standard_image_path(argc, argv, hooks.default_image);
    };
    spec.make_model = make_model;
    spec.handle_output = [hooks, model_id](const ImageInput &, const OUTPUT &out, const std::string &image_path) {
        if (!hooks.handle_output) {
            return;
        }
        const cv::Mat src = cv::imread(image_path, cv::IMREAD_COLOR);
        hooks.handle_output(src, out, image_path, model_id);
    };
    return run_benchmark(argc, argv, spec);
}

} // namespace benchmark
} // namespace apps
} // namespace jinq

#endif // MORTRED_APPS_BENCHMARK_IMAGE_FAMILY_H
