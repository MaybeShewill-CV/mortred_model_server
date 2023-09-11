/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: trt_helper.h
 * Date: 23-9-7
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_TRT_HELPER_H
#define MORTRED_MODEL_SERVER_TRT_HELPER_H

#include <string>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "TensorRT-8.6.1.6/NvInfer.h"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace trt_helper {

class EngineBinding {
  public:
    /***
     *
     */
    EngineBinding() noexcept = default;

    /***
     *
     */
    ~EngineBinding() noexcept = default;

    /***
     *
     * @param other
     */
    void swap(EngineBinding& other);

    /***
     *
     * @return
     */
    const int& index() const {
        return _m_index;
    }

    /***
     *
     * @param index
     */
    void set_index(int index) {
        _m_index = index;
    }

    /***
     *
     * @return
     */
    const std::string& name() const {
        return _m_name;
    }

    /***
     *
     * @param name
     */
    void set_name(const std::string& name) {
        _m_name = name;
    }

    /***
     *
     * @return
     */
    const nvinfer1::Dims& dims() const {
        return _m_dims;
    }

    /***
     *
     * @param dims
     */
    void set_dims(const nvinfer1::Dims& dims) {
        _m_dims = dims;
    }

    /***
     *
     * @return
     */
    const int& volume() const {
        return _m_volume;
    }

    /***
     *
     * @param volume
     */
    void set_volume(int volume) {
        _m_volume = volume;
    }

    /***
     *
     * @return
     */
    bool is_dynamic() const {
        for (int i = 0; i < _m_dims.nbDims; ++i) {
            if (_m_dims.d[i] == -1) {
                return true;
            }
        }
        return false;
    }

    /***
     *
     * @return
     */
    const bool& is_input() const {
        return _m_is_input;
    }

    /***
     *
     * @param val
     */
    void set_is_input(bool val) {
        _m_is_input = val;
    }

    /***
     *
     * @return
     */
    std::string info_str() const {
        std::string dims_str  = "(";
        for (int32_t i = 0; i < _m_dims.nbDims; ++i) {
            dims_str += std::to_string(_m_dims.d[i]);
            if (i < _m_dims.nbDims - 1) {
                dims_str.push_back(',');
            }
        }
        dims_str.push_back(')');
        std::string out = "name: '" + _m_name + "'"
                          + " ;  dims: " + dims_str
                          + " ;  isInput: " + (_m_is_input ? "true" : "false")
                          + " ;  dynamic: " + (is_dynamic() ? "true" : "false");

        return out;
    }

  private:
    int _m_index = 0;
    std::string _m_name;
    nvinfer1::Dims _m_dims{};
    int _m_volume  = 0;
    bool _m_is_input = false;
};

class DeviceMemory {
  public:
    /***
     *
     */
    DeviceMemory() noexcept = default;

    /***
     *
     */
    ~DeviceMemory() noexcept {
        for (auto& ptr : _m_memory) {
            auto cuda_status = cudaFree(ptr);
            if (cuda_status != cudaSuccess) {
                LOG(ERROR) << "free cuda memory failed code str: " << cudaGetErrorString(cuda_status);
                break;
            }
        }
        DLOG(INFO) << "~destruct device memory";
    };

  private:
    DeviceMemory(const DeviceMemory&) = default;
  public:

    void swap(DeviceMemory& other) {
        std::swap(_m_memory, other._m_memory);
    }

    /***
     *
     * @return
     */
    void** begin() const {
        return (void**)_m_memory.data();
    }

    /***
     *
     * @return
     */
    void** back() {
        return (void**)(_m_memory.data() + _m_memory.size());
    }

    /***
     *
     * @param index
     * @return
     */
    void* at(const int& index) const {
        return _m_memory[index];
    }

    /***
     *
     * @param data
     */
    void push_back(void* data) {
        _m_memory.push_back(data);
    }

  private:
    std::vector<void*> _m_memory;
};

class TrtLogger : public nvinfer1::ILogger {
  public:
    TrtLogger() = default;

    ~TrtLogger() override = default;

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity == nvinfer1::ILogger::Severity::kINFO) {
            DLOG(INFO) << msg;
        } else if (severity == nvinfer1::ILogger::Severity::kERROR) {
            LOG(ERROR) << msg;
        } else if (severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR) {
            LOG(ERROR) << msg;
        } else if (severity == nvinfer1::ILogger::Severity::kWARNING) {
            DLOG(WARNING) << msg;
        } else if (severity == nvinfer1::ILogger::Severity::kVERBOSE) {
            VLOG(1) << msg;
        } else {
            LOG(FATAL) << msg;
        }
    }
};

class TrtHelper {
  public:
    TrtHelper() = delete;
    ~TrtHelper() = delete;
    TrtHelper(const TrtHelper& transformer) = delete;
    TrtHelper& operator=(const TrtHelper& transformer) = delete;

    static std::string dims_to_string(const nvinfer1::Dims& dims) {
        std::string out = "(";
        for (int32_t i = 0; i < dims.nbDims; ++i) {
            out += std::to_string(dims.d[i]);
            if (i < dims.nbDims - 1) {
                out.push_back(',');
            }
        }
        out.push_back(')');
        return out;
    }

    /***
     *
     * @param dims
     * @return
     */
    static int32_t dims_volume(const nvinfer1::Dims& dims) {
        int32_t r = 0;
        if (dims.nbDims > 0) {
            r = 1;
        }

        for (int32_t i = 0; i < dims.nbDims; ++i) {
            r = r * dims.d[i];
        }
        return r;
    }

    /***
    *
    * @param engine
    * @param name
    * @param binding
    * @return
     */
    static bool setup_engine_binding(
        const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
        const std::string& name, EngineBinding& binding);

    /***
     *
     * @param engine
     * @param index
     * @param binding
     * @return
     */
    static bool setup_engine_binding(
        const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
        const int& index, EngineBinding& binding);

    /***
     *
     * @param engine
     * @param output
     * @return
     */
    static common::StatusCode setup_device_memory(
        std::unique_ptr<nvinfer1::ICudaEngine>& engine,
        DeviceMemory& output);

    /***
     *
     * @param engine
     * @param context
     * @param output
     * @return
     */
    static common::StatusCode setup_device_memory(
        std::unique_ptr<nvinfer1::ICudaEngine>& engine,
        std::unique_ptr<nvinfer1::IExecutionContext>& context,
        DeviceMemory& output);

    /***
     *
     * @return
     */
    static bool opencv_has_cuda() {
        auto r = cv::cuda::getCudaEnabledDeviceCount();
        return r > 0;
    }

    /***
     *
     * @param engine
     */
    static void print_bindings(const std::unique_ptr<nvinfer1::ICudaEngine>& engine) {
        const int32_t n_bindings = engine->getNbBindings();
        for(int i = 0; i < n_bindings; ++i) {
            EngineBinding binding;
            setup_engine_binding(engine, i, binding);
            auto binding_info = binding.info_str();
            LOG(INFO) << "binding: " << i << ", info: " << binding_info;
        }
    }
};
}
}
}

#endif // MORTRED_MODEL_SERVER_TRT_HELPER_H
