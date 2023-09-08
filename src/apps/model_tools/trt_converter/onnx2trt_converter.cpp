/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: onnx2trt_converter_tool.cpp
* Date: 23-8-24
************************************************/
// convert onnx model file into trt engine file

#include <string>
#include <chrono>

#include <glog/logging.h>

#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/trt_helper//onnx2trt_model_builder.h"

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::trt_helper::Onnx2TrtModelBuilder;
using jinq::models::trt_helper::TRT_PRECISION_MODE;

int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);

    if (argc < 3) {
        LOG(INFO) << "wrong usage";
        LOG(INFO) << "exe onnx_file_path out_engine_file_path [fp_mode]";
        return -1;
    }

    std::string onnx_file_path = argv[1];
    if (!FilePathUtil::is_file_exist(onnx_file_path)) {
        LOG(ERROR) << "onnx file: " << onnx_file_path << " not exists";
        return -1;
    }
    std::string engine_file_path = argv[2];

    auto fp_mode = TRT_PRECISION_MODE::TRT_PRECISION_FP32;
    if (argc >= 4) {
        fp_mode = static_cast<TRT_PRECISION_MODE>(std::stoi(argv[3]));
    }

    Onnx2TrtModelBuilder converter;
    auto t_start = std::chrono::system_clock::now();
    auto status = converter.build_engine_file(
            onnx_file_path, engine_file_path, fp_mode);
    auto t_end = std::chrono::system_clock::now();
    auto t_cost = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count();
    if (status != StatusCode::OK) {
        LOG(ERROR) << "convert onnx file: " << onnx_file_path << " into trt engine file failed, status code: " << status;
        return -1;
    } else {
        LOG(INFO) << "successfully convert onnx file into trt engine, cost time: " << t_cost << "s";
        return 1;
    }
}