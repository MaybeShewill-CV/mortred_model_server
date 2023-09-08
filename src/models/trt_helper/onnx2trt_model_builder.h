/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: onnx2trt_model_builder.h
* Date: 23-8-23
************************************************/

#ifndef MORTRED_MODEL_SERVER_ONNX2TRT_MODEL_BUILDER_H
#define MORTRED_MODEL_SERVER_ONNX2TRT_MODEL_BUILDER_H

#include <memory>
#include <string>

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace trt_helper {

enum TRT_PRECISION_MODE {
    TRT_PRECISION_FP32 = 0,
    TRT_PRECISION_FP16 = 1,
};

class Onnx2TrtModelBuilder {
public:

    /***
    * 构造函数
    * @param config
    */
    Onnx2TrtModelBuilder();

    /***
     *
     */
    ~Onnx2TrtModelBuilder();

    /***
    * 赋值构造函数
    * @param transformer
    */
    Onnx2TrtModelBuilder(const Onnx2TrtModelBuilder& transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    Onnx2TrtModelBuilder& operator=(const Onnx2TrtModelBuilder& transformer) = delete;

    /***
     *
     * @param input_onnx_file_path
     * @param output_engine_file_path
     * @param fp_mode
     * @return
     */
    jinq::common::StatusCode build_engine_file(
        const std::string& input_onnx_file_path,
        const std::string& output_engine_file_path,
        TRT_PRECISION_MODE fp_mode = TRT_PRECISION_MODE::TRT_PRECISION_FP32);

private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
};
}
}

#endif //MORTRED_MODEL_SERVER_ONNX2TRT_MODEL_BUILDER_H
