/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: onnx2trt_model_builder.cpp
* Date: 23-8-23
************************************************/

#include "onnx2trt_model_builder.h"

#include <fstream>

#include "glog/logging.h"
#include "TensorRT-8.6.1.6/NvOnnxParser.h"

#include "common/file_path_util.h"
#include "models/trt_helper/trt_helper.h"

namespace jinq {
namespace models {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;
using jinq::models::trt_helper::TrtLogger;

namespace trt_helper {

class Onnx2TrtModelBuilder::Impl {
public:
    /***
     *
     */
    Impl();

    /***
     *
     */
    ~Impl() = default;

    /***
    *
    * @param transformer
    */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;



    /***
     *
     * @param input_onnx_file_path
     * @param output_engine_file_path
     * @param fp_mode
     * @return
     */
    StatusCode build_engine_file(
        const std::string& input_onnx_file_path,
        const std::string& output_engine_file_path,
        TRT_PRECISION_MODE fp_mode = TRT_PRECISION_MODE::TRT_PRECISION_FP32);

private:
    std::unique_ptr<TrtLogger> _m_trt_logger;

    /***
     *
     * @param inputFilePath
     * @param output
     * @param precision
     * @return
     */
    StatusCode build_engine(
            const std::string& input_file_path,
            std::shared_ptr<nvinfer1::IHostMemory>* output,
            TRT_PRECISION_MODE precision);
};

/***
 *
 */
Onnx2TrtModelBuilder::Impl::Impl() {
    _m_trt_logger = std::make_unique<TrtLogger>();
}

/***
 *
 * @param input_onnx_file_path
 * @param output_engine_file_path
 * @param fp_mode
 * @return
 */
StatusCode Onnx2TrtModelBuilder::Impl::build_engine_file(
        const std::string &input_onnx_file_path,
        const std::string &output_engine_file_path,
        jinq::models::trt_helper::TRT_PRECISION_MODE fp_mode) {
    // convert engine file out
    std::shared_ptr<nvinfer1::IHostMemory> engine_out;
    auto status = build_engine(input_onnx_file_path, &engine_out, fp_mode);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "build engine file from onnx file failed";
        return StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED;
    }

    // write output into local engine file
    std::ofstream out_engine_file;
    out_engine_file.open(output_engine_file_path, std::ios::out | std::ios::binary);
    out_engine_file.write((char*)engine_out->data(), static_cast<long>(engine_out->size()));
    if(!out_engine_file.good()) {
        LOG(ERROR) << "write converted onnx file into engine file failed";
        return StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED;
    }
    out_engine_file.close();

    return StatusCode::OK;
}

/***
 *
 * @param inputFilePath
 * @param output
 * @param precision
 * @return
 */
StatusCode Onnx2TrtModelBuilder::Impl::build_engine(
    const std::string &input_file_path,
    std::shared_ptr<nvinfer1::IHostMemory> *output,
    jinq::models::trt_helper::TRT_PRECISION_MODE precision) {
    // construct infer builder
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*_m_trt_logger));
    const auto explicit_batch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));

    builder->setMaxBatchSize(1);

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, *_m_trt_logger));
    if(!parser->parseFromFile(input_file_path.c_str(),(int)nvinfer1::ILogger::Severity::kWARNING)) {
        LOG(ERROR) << "could not parse ONNX model from file: " << input_file_path;
        return StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED;
    }

    // set config
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    size_t workspace = 1 << 30;
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace * 6);
    if (precision == TRT_PRECISION_MODE::TRT_PRECISION_FP32) {
        // skip
    } else if (precision == TRT_PRECISION_MODE::TRT_PRECISION_FP16) {
        if (!builder->platformHasFastFp16()) {
            LOG(ERROR) << "build engine file from onnx failed, fp16 precision specified, "
                          "but not supported by current platform";
            return StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED;
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        LOG(ERROR) << "not supported fp mode of: " << precision;
        return StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED;
    }
    LOG(INFO) << "start building and serializing engine, coffee time ...";
    std::shared_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (nullptr == serialized) {
        LOG(ERROR) << "could not build serialized engine";
        return StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED;
    }
    *output = serialized;
    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
Onnx2TrtModelBuilder::Onnx2TrtModelBuilder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
Onnx2TrtModelBuilder::~Onnx2TrtModelBuilder() = default;

/***
 *
 * @tparam INPUT
 * @tparam OUTPUT
 * @param cfg
 * @return
 */
StatusCode Onnx2TrtModelBuilder::build_engine_file(
    const std::string& input_onnx_file_path,
    const std::string& output_engine_file_path,
    jinq::models::trt_helper::TRT_PRECISION_MODE fp_mode) {
    return _m_pimpl->build_engine_file(input_onnx_file_path, output_engine_file_path, fp_mode);
}
}
}
}