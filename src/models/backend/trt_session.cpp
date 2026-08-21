/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: trt_session.cpp
 * Date: 26-8-20
 ************************************************/

#include "models/backend/trt_session.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "NvInferPlugin.h"
#include "NvInferVersion.h"
#include "glog/logging.h"

#include "common/file_path_util.h"

namespace jinq {
namespace models {
namespace backend {

using jinq::common::StatusCode;

namespace {

nvinfer1::Dims to_trt_dims(const std::vector<int64_t>& shape) {
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        dims.d[idx] = static_cast<int64_t>(shape[idx]);
    }
    return dims;
}

std::vector<int64_t> from_trt_dims(const nvinfer1::Dims& dims) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(dims.nbDims));
    for (int32_t idx = 0; idx < dims.nbDims; ++idx) {
        shape.push_back(dims.d[idx]);
    }
    return shape;
}

bool dtype_from_trt(const nvinfer1::DataType& dtype, DType* out, std::string* err) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            *out = DType::F32;
            return true;
        case nvinfer1::DataType::kINT32:
            *out = DType::I32;
            return true;
        case nvinfer1::DataType::kUINT8:
            *out = DType::U8;
            return true;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64:
            *out = DType::I64;
            return true;
#endif
        default:
            if (err != nullptr) {
                *err = "unsupported tensorrt io dtype: " + std::to_string(static_cast<int>(dtype));
            }
            return false;
    }
}

}  // namespace

namespace trt_detail {

void SessionLogger::log(Severity severity, const char* msg) noexcept {
    if (msg == nullptr) {
        return;
    }
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            LOG(ERROR) << "[tensorrt] " << msg;
            break;
        case Severity::kWARNING:
            LOG(WARNING) << "[tensorrt] " << msg;
            break;
        case Severity::kINFO:
            LOG(INFO) << "[tensorrt] " << msg;
            break;
        default:
            break;
    }
}

}  // namespace trt_detail

/***
 * TensorRT calls this allocator when an output shape depends on runtime data
 * (for example the lightglue extractor's NMS output).  It owns the device
 * allocation and records the shape communicated through notifyShape.
 */
class TrtSession::DynamicOutputAllocator final : public nvinfer1::IOutputAllocator {
  public:
    DynamicOutputAllocator() = default;
    ~DynamicOutputAllocator() override {
        release();
    }

    void* reallocateOutput(char const* tensor_name, void* current_memory, uint64_t size,
                           uint64_t alignment) noexcept override {
        (void)tensor_name;
        (void)current_memory;
        (void)alignment;
        allocation_failed = false;
        if (size == 0) {
            allocation_failed = true;
            return nullptr;
        }
        if (memory != nullptr && allocated_bytes >= size) {
            return memory;
        }
        release();
        const auto status = cudaMalloc(&memory, size);
        if (status != cudaSuccess || memory == nullptr) {
            LOG(ERROR) << "allocate dynamic tensorrt output failed: "
                       << cudaGetErrorString(status);
            memory = nullptr;
            allocation_failed = true;
            return nullptr;
        }
        allocated_bytes = size;
        return memory;
    }

    void notifyShape(char const* tensor_name, nvinfer1::Dims const& dims) noexcept override {
        (void)tensor_name;
        shape = dims;
        shape_notified = true;
    }

    void reset_for_run() {
        shape = nvinfer1::Dims{};
        shape_notified = false;
        allocation_failed = false;
    }

    bool failed() const {
        return allocation_failed;
    }

    bool has_shape() const {
        return shape_notified;
    }

    std::vector<int64_t> reported_shape() const {
        return from_trt_dims(shape);
    }

    size_t capacity() const {
        return allocated_bytes;
    }

    void* data() const {
        return memory;
    }

  private:
    void release() {
        if (memory != nullptr) {
            const auto status = cudaFree(memory);
            if (status != cudaSuccess) {
                LOG(ERROR) << "free dynamic tensorrt output failed: "
                           << cudaGetErrorString(status);
            }
            memory = nullptr;
        }
        allocated_bytes = 0;
    }

    void* memory = nullptr;
    size_t allocated_bytes = 0;
    nvinfer1::Dims shape{};
    bool shape_notified = false;
    bool allocation_failed = false;
};

TrtSession::DeviceBuffer& TrtSession::DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (memory != nullptr) {
            cudaFree(memory);
        }
        memory = other.memory;
        bytes = other.bytes;
        other.memory = nullptr;
        other.bytes = 0;
    }
    return *this;
}

TrtSession::DeviceBuffer::~DeviceBuffer() {
    if (memory != nullptr) {
        const auto status = cudaFree(memory);
        if (status != cudaSuccess) {
            LOG(ERROR) << "free tensorrt device buffer failed: " << cudaGetErrorString(status);
        }
        memory = nullptr;
    }
}

StatusCode TrtSession::DeviceBuffer::ensure(size_t size_bytes) {
    if (size_bytes == 0) {
        return StatusCode::TRT_ALLOC_MEMO_FAILED;
    }
    if (memory != nullptr && bytes >= size_bytes) {
        return StatusCode::OK;
    }
    if (memory != nullptr) {
        const auto status = cudaFree(memory);
        if (status != cudaSuccess) {
            LOG(ERROR) << "free tensorrt device buffer failed: " << cudaGetErrorString(status);
            memory = nullptr;
            return StatusCode::TRT_ALLOC_MEMO_FAILED;
        }
        memory = nullptr;
        bytes = 0;
    }
    void* device_memory = nullptr;
    const auto status = cudaMalloc(&device_memory, size_bytes);
    if (status != cudaSuccess || device_memory == nullptr) {
        LOG(ERROR) << "allocate tensorrt device buffer failed: " << cudaGetErrorString(status);
        return StatusCode::TRT_ALLOC_MEMO_FAILED;
    }
    memory = device_memory;
    bytes = size_bytes;
    return StatusCode::OK;
}

TrtSession::~TrtSession() {
    _m_device_buffers.clear();
    if (_m_stream != nullptr) {
        cudaStreamDestroy(_m_stream);
        _m_stream = nullptr;
    }
    // TRT 10 removed destroy(); public destructors are the supported path on
    // both 8.6 and 10.x
    delete _m_context;
    _m_context = nullptr;
    delete _m_engine;
    _m_engine = nullptr;
    delete _m_runtime;
    _m_runtime = nullptr;
    for (auto& item : _m_output_allocators) {
        delete item.second;
    }
    _m_output_allocators.clear();
}

StatusCode TrtSession::init(const BackendConfig& config, std::string* err) {
    if (err != nullptr) {
        err->clear();
    }
    if (!jinq::common::FilePathUtil::is_file_exist(config.model_file_path)) {
        if (err != nullptr) {
            *err = "tensorrt engine file not exist: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }

    std::ifstream engine_file(config.model_file_path, std::ios_base::in | std::ios_base::binary);
    if (!engine_file) {
        if (err != nullptr) {
            *err = "read tensorrt engine failed: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }
    std::stringstream buffer;
    buffer << engine_file.rdbuf();
    const std::string engine_stream = buffer.str();
    if (engine_stream.empty()) {
        if (err != nullptr) {
            *err = "tensorrt engine file is empty: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (!initLibNvInferPlugins(nullptr, "")) {
        if (err != nullptr) {
            *err = "init tensorrt plugin registry failed";
        }
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_runtime = nvinfer1::createInferRuntime(_m_logger);
    if (_m_runtime == nullptr) {
        if (err != nullptr) {
            *err = "create tensorrt runtime failed";
        }
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_engine = _m_runtime->deserializeCudaEngine(engine_stream.data(), engine_stream.size());
    if (_m_engine == nullptr) {
        if (err != nullptr) {
            *err = "deserialize tensorrt engine failed: " + config.model_file_path
                   + " (engine was built by an incompatible tensorrt version)";
        }
        delete _m_runtime;
        _m_runtime = nullptr;
        return StatusCode::MODEL_INIT_FAILED;
    }
    _m_context = _m_engine->createExecutionContext();
    if (_m_context == nullptr) {
        if (err != nullptr) {
            *err = "create tensorrt execution context failed";
        }
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto is_input = [this](const std::string& name) {
        return _m_engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    };
    const auto wanted_input = [&config, &is_input](const std::string& name) {
        if (!is_input(name)) {
            return false;
        }
        if (config.input_names.empty()) {
            return true;
        }
        return std::find(config.input_names.begin(), config.input_names.end(), name) !=
               config.input_names.end();
    };
    const auto wanted_output = [this, &config](const std::string& name) {
        if (_m_engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) {
            return false;
        }
        if (config.output_names.empty()) {
            return true;
        }
        return std::find(config.output_names.begin(), config.output_names.end(), name) !=
               config.output_names.end();
    };

    const int32_t io_count = _m_engine->getNbIOTensors();
    for (int32_t idx = 0; idx < io_count; ++idx) {
        const char* raw_name = _m_engine->getIOTensorName(idx);
        if (raw_name == nullptr) {
            continue;
        }
        const std::string name(raw_name);
        const bool as_input = wanted_input(name);
        const bool as_output = wanted_output(name);
        if (!as_input && !as_output) {
            continue;
        }
        TensorInfo info;
        info.name = name;
        std::string dtype_err;
        if (!dtype_from_trt(_m_engine->getTensorDataType(name.c_str()), &info.dtype, &dtype_err)) {
            if (err != nullptr) {
                *err = "tensorrt io '" + name + "': " + dtype_err;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
        info.shape = from_trt_dims(_m_engine->getTensorShape(name.c_str()));
        info.dynamic = shape_is_dynamic(info.shape);
        _m_device_buffers.emplace(name, DeviceBuffer{});
        if (as_input) {
            _m_input_infos.push_back(std::move(info));
        } else {
            _m_output_infos.push_back(std::move(info));
        }
    }
    if (_m_input_infos.empty() || _m_output_infos.empty()) {
        if (err != nullptr) {
            *err = "tensorrt engine exposes no io tensors: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }
    for (const auto& name : config.input_names) {
        const auto found = std::any_of(
            _m_input_infos.begin(), _m_input_infos.end(),
            [&name](const TensorInfo& info) { return info.name == name; });
        if (!found) {
            if (err != nullptr) {
                *err = "configured tensorrt input tensor not found: " + name;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
    }
    for (const auto& name : config.output_names) {
        const auto found = std::any_of(
            _m_output_infos.begin(), _m_output_infos.end(),
            [&name](const TensorInfo& info) { return info.name == name; });
        if (!found) {
            if (err != nullptr) {
                *err = "configured tensorrt output tensor not found: " + name;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
    }

    // Outputs whose shape cannot be inferred from the input shape require an
    // IOutputAllocator. Without one, getTensorShape() remains dynamic and there
    // is no concrete output address to bind before enqueueV3().
    for (const auto& info : _m_output_infos) {
        if (!info.dynamic) {
            continue;
        }
        auto* allocator = new DynamicOutputAllocator();
        _m_output_allocators.emplace(info.name, allocator);
        const bool allocator_ready =
            _m_context->setOutputAllocator(info.name.c_str(), allocator) &&
            _m_context->setTensorAddress(info.name.c_str(), nullptr);
        if (!allocator_ready) {
            if (err != nullptr) {
                *err = "configure tensorrt output allocator failed: " + info.name;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
    }

    const auto cuda_status = cudaStreamCreate(&_m_stream);
    if (cuda_status != cudaSuccess) {
        if (err != nullptr) {
            *err = std::string("create cuda stream failed: ") + cudaGetErrorString(cuda_status);
        }
        return StatusCode::TRT_CUDA_ERROR;
    }

    _m_model_file_path = config.model_file_path;
    for (const auto& info : _m_input_infos) {
        LOG(INFO) << "tensorrt session input: " << info.to_string();
    }
    for (const auto& info : _m_output_infos) {
        LOG(INFO) << "tensorrt session output: " << info.to_string();
    }
    return StatusCode::OK;
}

StatusCode TrtSession::run(const std::vector<NamedTensor>& inputs,
                           std::vector<NamedTensor>& outputs) {
    if (_m_context == nullptr || _m_engine == nullptr) {
        LOG(ERROR) << "tensorrt session is not initialized";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (inputs.size() != _m_input_infos.size()) {
        LOG(ERROR) << "tensorrt session expects " << _m_input_infos.size() << " inputs, got "
                   << inputs.size();
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    // apply input shapes (also validates every provided input against the io)
    for (const auto& named : inputs) {
        const auto info_iter = std::find_if(
            _m_input_infos.begin(), _m_input_infos.end(),
            [&named](const TensorInfo& info) { return info.name == named.name; });
        if (info_iter == _m_input_infos.end()) {
            LOG(ERROR) << "unknown tensorrt input tensor: " << named.name;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        if (info_iter->dtype != named.tensor.dtype) {
            LOG(ERROR) << "tensorrt input '" << named.name << "' dtype mismatch, expected "
                       << info_iter->to_string() << ", got " << dtype_to_string(named.tensor.dtype);
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        if (!named.tensor.shape_is_concrete()) {
            LOG(ERROR) << "tensorrt input '" << named.name << "' has non concrete shape "
                       << shape_to_string(named.tensor.shape);
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        if (!_m_context->setInputShape(named.name.c_str(),
                                        to_trt_dims(named.tensor.shape))) {
            LOG(ERROR) << "tensorrt input '" << named.name << "' set shape "
                       << shape_to_string(named.tensor.shape)
                       << " rejected (outside the optimization profile?)";
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
    }

    // H2D + bind inputs
    for (const auto& named : inputs) {
        auto& buffer = _m_device_buffers.at(named.name);
        const auto status = buffer.ensure(
            static_cast<size_t>(named.tensor.element_count()) * dtype_size(named.tensor.dtype));
        if (status != StatusCode::OK) {
            return status;
        }
        const auto cuda_status = cudaMemcpyAsync(
            buffer.memory, named.tensor.buffer.data(), named.tensor.byte_size(),
            cudaMemcpyHostToDevice, _m_stream);
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "tensorrt H2D copy failed for '" << named.name
                       << "': " << cudaGetErrorString(cuda_status);
            return StatusCode::TRT_CUDA_ERROR;
        }
        if (!_m_context->setTensorAddress(named.name.c_str(), buffer.memory)) {
            LOG(ERROR) << "tensorrt set input tensor address failed: " << named.name;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
    }

    // Resolve and bind outputs whose shape is inferable from the inputs.
    // Runtime-data-dependent outputs are left to their IOutputAllocator.
    struct ResolvedOutput {
        const TensorInfo* info;
        std::vector<int64_t> shape;
        void* memory;
    };
    std::vector<ResolvedOutput> resolved_outputs;
    resolved_outputs.reserve(_m_output_infos.size());
    for (const auto& info : _m_output_infos) {
        resolved_outputs.push_back({&info, {}, nullptr});
    }
    for (auto& item : _m_output_allocators) {
        item.second->reset_for_run();
    }
    for (size_t idx = 0; idx < _m_output_infos.size(); ++idx) {
        const auto& info = _m_output_infos[idx];
        const auto allocator_iter = _m_output_allocators.find(info.name);
        if (allocator_iter != _m_output_allocators.end()) {
            continue;
        }
        auto shape = from_trt_dims(_m_context->getTensorShape(info.name.c_str()));
        if (shape_is_dynamic(shape)) {
            LOG(ERROR) << "tensorrt output '" << info.name << "' shape unresolved after input "
                       << "shapes: " << shape_to_string(shape);
            return StatusCode::TRT_ALLOC_DYNAMIC_SHAPE_MEMO;
        }
        auto& buffer = _m_device_buffers.at(info.name);
        const auto bytes =
            static_cast<size_t>(shape_volume(shape)) * dtype_size(info.dtype);
        const auto status = buffer.ensure(bytes);
        if (status != StatusCode::OK) {
            return status;
        }
        if (!_m_context->setTensorAddress(info.name.c_str(), buffer.memory)) {
            LOG(ERROR) << "tensorrt set output tensor address failed: " << info.name;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        resolved_outputs[idx] = {&info, std::move(shape), buffer.memory};
    }

    if (!_m_context->enqueueV3(_m_stream)) {
        LOG(ERROR) << "tensorrt enqueueV3 inference failed";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    const auto sync_status = cudaStreamSynchronize(_m_stream);
    if (sync_status != cudaSuccess) {
        LOG(ERROR) << "tensorrt stream sync failed: " << cudaGetErrorString(sync_status);
        return StatusCode::TRT_CUDA_ERROR;
    }

    for (size_t idx = 0; idx < _m_output_infos.size(); ++idx) {
        const auto& info = _m_output_infos[idx];
        const auto allocator_iter = _m_output_allocators.find(info.name);
        if (allocator_iter == _m_output_allocators.end()) {
            continue;
        }
        const auto* allocator = allocator_iter->second;
        if (allocator->failed() || allocator->data() == nullptr) {
            LOG(ERROR) << "tensorrt dynamic output '" << info.name << "' allocation failed";
            return StatusCode::TRT_ALLOC_DYNAMIC_SHAPE_MEMO;
        }
        auto shape = allocator->reported_shape();
        if (!allocator->has_shape() || shape_is_dynamic(shape)) {
            shape = from_trt_dims(_m_context->getTensorShape(info.name.c_str()));
        }
        if (shape_is_dynamic(shape)) {
            LOG(ERROR) << "tensorrt dynamic output '" << info.name
                       << "' shape unresolved after inference: " << shape_to_string(shape);
            return StatusCode::TRT_ALLOC_DYNAMIC_SHAPE_MEMO;
        }
        const auto bytes =
            static_cast<size_t>(shape_volume(shape)) * dtype_size(info.dtype);
        if (bytes == 0 || allocator->capacity() < bytes) {
            LOG(ERROR) << "tensorrt dynamic output '" << info.name << "' allocation "
                       << allocator->capacity() << " bytes is smaller than output "
                       << bytes << " bytes";
            return StatusCode::TRT_ALLOC_DYNAMIC_SHAPE_MEMO;
        }
        resolved_outputs[idx] = {&info, std::move(shape), allocator->data()};
    }

    outputs.clear();
    outputs.reserve(resolved_outputs.size());
    for (const auto& item : resolved_outputs) {
        if (item.memory == nullptr) {
            LOG(ERROR) << "tensorrt output '" << item.info->name
                       << "' was not bound to device memory";
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        const auto& info = *item.info;
        const auto& shape = item.shape;
        NamedTensor named;
        named.name = info.name;
        named.tensor.dtype = info.dtype;
        named.tensor.shape = shape;
        const auto bytes =
            static_cast<size_t>(shape_volume(shape)) * dtype_size(info.dtype);
        named.tensor.buffer.resize(bytes);
        const auto cuda_status = cudaMemcpyAsync(
            named.tensor.buffer.data(), item.memory, bytes, cudaMemcpyDeviceToHost, _m_stream);
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "tensorrt D2H copy failed for '" << info.name
                       << "': " << cudaGetErrorString(cuda_status);
            return StatusCode::TRT_CUDA_ERROR;
        }
        outputs.push_back(std::move(named));
    }
    const auto final_sync = cudaStreamSynchronize(_m_stream);
    if (final_sync != cudaSuccess) {
        LOG(ERROR) << "tensorrt stream sync failed: " << cudaGetErrorString(final_sync);
        return StatusCode::TRT_CUDA_ERROR;
    }
    return StatusCode::OK;
}

}  // namespace backend
}  // namespace models
}  // namespace jinq
