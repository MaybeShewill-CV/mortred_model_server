/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: trt_helper.cpp
 * Date: 23-9-7
 ************************************************/

#include "trt_helper.h"

#include <cstring>

#include <glog/logging.h>
#include <cuda_runtime_api.h>

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace trt_helper {

using jinq::common::StatusCode;

/*********** EngineBinding Implementations **********/
/***
 *
 * @param other
 */
void EngineBinding::swap(EngineBinding& other) {
    std::swap(_m_index, other._m_index);
    std::swap(_m_name, other._m_name);
    std::swap(_m_volume, other._m_volume);

    nvinfer1::Dims tmp;
    std::memcpy(&tmp, &_m_dims, sizeof(nvinfer1::Dims));
    std::memcpy(&_m_dims, &other._m_dims, sizeof(nvinfer1::Dims));
    std::memcpy(&other._m_dims, &tmp, sizeof(nvinfer1::Dims));

    std::swap(_m_is_input, other._m_is_input);
}

/******* Trt Helper Func Implementation *****/
/***
 *
 * @param engine
 * @param index
 * @param binding
 * @return
 */
bool TrtHelper::setup_engine_binding(
    const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
    const int& index, EngineBinding& binding) {

    binding.set_index(index);
    const char* name = engine->getIOTensorName(index);
    if (name == nullptr) {
        return false;
    }
    binding.set_name(std::string(name));
    binding.set_dims(engine->getTensorShape(name));
    binding.set_volume(dims_volume(binding.dims()));
    auto tensor_io_mode = engine->getTensorIOMode(name);
    if (tensor_io_mode == nvinfer1::TensorIOMode::kINPUT) {
        binding.set_is_input(true);
    } else {
        binding.set_is_input(false);
    }

    return true;
}

/***
 *
 * @param engine
 * @param name
 * @param binding
 * @return
 */
bool TrtHelper::setup_engine_binding(
    const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
    const std::string& name, EngineBinding& binding) {

    binding.set_name(name);
    binding.set_index(engine->getBindingIndex(name.c_str()));
    if (binding.index() == -1) {
        return false;
    }
    binding.set_dims(engine->getTensorShape(name.c_str()));
    binding.set_volume(dims_volume(binding.dims()));
    auto tensor_io_mode = engine->getTensorIOMode(name.c_str());
    if (tensor_io_mode == nvinfer1::TensorIOMode::kINPUT) {
        binding.set_is_input(true);
    } else {
        binding.set_is_input(false);
    }

    return true;
}

/***
 *
 * @param engine
 * @param output
 * @return
 */
StatusCode TrtHelper::setup_device_memory(std::unique_ptr<nvinfer1::ICudaEngine>& engine, DeviceMemory& output) {
    auto nb_bindings = engine->getNbIOTensors();
    for (int i = 0; i < nb_bindings; ++i) {
        auto tensorname = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(tensorname);
        auto volume = dims_volume(dims);
        DLOG(INFO) << "tensor: " << tensorname <<  ", volume: " << volume << ", dims: " << dims_to_string(dims);
        void* cuda_memo = nullptr;
        auto r = cudaMalloc(&cuda_memo, volume * sizeof(float));
        if (r != 0 || cuda_memo == nullptr) {
            LOG(ERROR) << "Setup device memory failed error str: " << cudaGetErrorString(r);
            return StatusCode::TRT_ALLOC_MEMO_FAILED;
        }
        output.push_back(cuda_memo);
    }
    return StatusCode::OK;
}

/***
 *
 * @param engine
 * @param context
 * @param output
 * @return
 */
StatusCode TrtHelper::setup_device_memory(
    std::unique_ptr<nvinfer1::ICudaEngine> &engine,
    std::unique_ptr<nvinfer1::IExecutionContext> &context,
    DeviceMemory &output) {
    auto nb_bindings = engine->getNbIOTensors();
    for (int i = 0; i < nb_bindings; ++i) {
        auto tensorname = engine->getIOTensorName(i);
        auto dims = context->getTensorShape(tensorname);
        auto volume = dims_volume(dims);
        DLOG(INFO) << "tensor: " << tensorname <<  ", volume: " << volume << ", dims: " << dims_to_string(dims);
        void* cuda_memo = nullptr;
        auto r = cudaMalloc(&cuda_memo, volume * sizeof(float));
        if (r != 0 || cuda_memo == nullptr) {
            LOG(ERROR) << "Setup device memory failed error str: " << cudaGetErrorString(r);
            return StatusCode::TRT_ALLOC_MEMO_FAILED;
        }
        output.push_back(cuda_memo);
    }
    return StatusCode::OK;
}

}
}
}