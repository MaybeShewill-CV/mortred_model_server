/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: backend/ort_session.cpp
 * Date: 2026-08-20
 ************************************************/

#include "models/backend/ort_session.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

#include "common/file_path_util.h"

namespace jinq {
namespace models {
namespace backend {

using jinq::common::StatusCode;

namespace {

ONNXTensorElementDataType to_ort_dtype(const DType& dtype) {
    switch (dtype) {
        case DType::F32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DType::I32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case DType::I64:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DType::U8:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

bool dtype_from_ort(const ONNXTensorElementDataType& dtype, DType* out, std::string* err) {
    switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            *out = DType::F32;
            return true;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            *out = DType::I32;
            return true;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            *out = DType::I64;
            return true;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            *out = DType::U8;
            return true;
        default:
            if (err != nullptr) {
                *err = "unsupported onnxruntime tensor element type: " + std::to_string(dtype);
            }
            return false;
    }
}

const void* ort_tensor_data(const Ort::Value& value, const DType& dtype) {
    switch (dtype) {
        case DType::F32:
            return value.GetTensorData<float>();
        case DType::I32:
            return value.GetTensorData<int32_t>();
        case DType::I64:
            return value.GetTensorData<int64_t>();
        case DType::U8:
            return value.GetTensorData<uint8_t>();
        default:
            return nullptr;
    }
}

}  // namespace

StatusCode OrtSession::init(const BackendConfig& config, std::string* err) {
    if (err != nullptr) {
        err->clear();
    }
    if (!jinq::common::FilePathUtil::is_file_exist(config.model_file_path)) {
        if (err != nullptr) {
            *err = "onnx model file not exist: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }

    try {
        _m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "mortred");
        _m_session_options = Ort::SessionOptions();
        _m_session_options.SetIntraOpNumThreads(config.threads);
        _m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        _m_session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        if (config.use_cuda()) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config.device_id;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1;
            cuda_options.do_copy_in_default_stream = 1;
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;
            _m_session_options.AppendExecutionProvider_CUDA(cuda_options);
            // fuse/transform passes of ALL are not guaranteed on the CUDA EP
            _m_session_options.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }
        _m_session = std::make_unique<Ort::Session>(
            _m_env, config.model_file_path.c_str(), _m_session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        const auto fill_infos = [&allocator](size_t count, auto get_name, auto get_type_info,
                                             std::vector<TensorInfo>* infos,
                                             std::vector<std::string>* names,
                                             std::vector<const char*>* name_ptrs,
                                             std::string* parse_err) -> bool {
            for (size_t idx = 0; idx < count; ++idx) {
                TensorInfo info;
                info.name = get_name(idx, allocator);
                // ConstTensorTypeAndShapeInfo borrows from the TypeInfo: keep
                // the owner alive while reading shape/element type
                const auto type_info = get_type_info(idx);
                const auto shape_and_type = type_info.GetTensorTypeAndShapeInfo();
                if (!dtype_from_ort(shape_and_type.GetElementType(), &info.dtype, parse_err)) {
                    return false;
                }
                info.shape = shape_and_type.GetShape();
                info.dynamic = shape_is_dynamic(info.shape);
                infos->push_back(std::move(info));
                names->push_back(infos->back().name);
            }
            name_ptrs->clear();
            name_ptrs->reserve(names->size());
            for (const auto& name : *names) {
                name_ptrs->push_back(name.c_str());
            }
            return true;
        };

        const bool inputs_ok = fill_infos(
            _m_session->GetInputCount(),
            [this](size_t idx, Ort::AllocatorWithDefaultOptions& alloc) {
                return std::string(_m_session->GetInputNameAllocated(idx, alloc).get());
            },
            [this](size_t idx) { return _m_session->GetInputTypeInfo(idx); }, &_m_input_infos,
            &_m_input_names, &_m_input_name_ptrs, err);
        if (!inputs_ok) {
            return StatusCode::MODEL_INIT_FAILED;
        }
        const bool outputs_ok = fill_infos(
            _m_session->GetOutputCount(),
            [this](size_t idx, Ort::AllocatorWithDefaultOptions& alloc) {
                return std::string(_m_session->GetOutputNameAllocated(idx, alloc).get());
            },
            [this](size_t idx) { return _m_session->GetOutputTypeInfo(idx); }, &_m_output_infos,
            &_m_output_names, &_m_output_name_ptrs, err);
        if (!outputs_ok) {
            return StatusCode::MODEL_INIT_FAILED;
        }
    } catch (const Ort::Exception& exception) {
        if (err != nullptr) {
            *err = "onnxruntime init failed: " + std::string(exception.what());
        }
        _m_session.reset();
        return StatusCode::MODEL_INIT_FAILED;
    }

    if (_m_input_infos.empty() || _m_output_infos.empty()) {
        if (err != nullptr) {
            *err = "onnx model exposes no io tensors: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (!config.input_names.empty()) {
        for (const auto& name : config.input_names) {
            const auto found = std::find_if(
                _m_input_infos.begin(), _m_input_infos.end(),
                [&name](const TensorInfo& info) { return info.name == name; });
            if (found == _m_input_infos.end()) {
                if (err != nullptr) {
                    *err = "configured onnx input tensor not found: " + name;
                }
                return StatusCode::MODEL_INIT_FAILED;
            }
        }
    }
    if (!config.output_names.empty()) {
        for (const auto& name : config.output_names) {
            const auto found = std::find_if(
                _m_output_infos.begin(), _m_output_infos.end(),
                [&name](const TensorInfo& info) { return info.name == name; });
            if (found == _m_output_infos.end()) {
                if (err != nullptr) {
                    *err = "configured onnx output tensor not found: " + name;
                }
                return StatusCode::MODEL_INIT_FAILED;
            }
        }
    }

    _m_model_file_path = config.model_file_path;
    for (const auto& info : _m_input_infos) {
        LOG(INFO) << "ort session input: " << info.to_string();
    }
    for (const auto& info : _m_output_infos) {
        LOG(INFO) << "ort session output: " << info.to_string();
    }
    return StatusCode::OK;
}

StatusCode OrtSession::run(const std::vector<NamedTensor>& inputs,
                           std::vector<NamedTensor>& outputs) {
    if (_m_session == nullptr) {
        LOG(ERROR) << "onnxruntime session is not initialized";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (inputs.size() != _m_input_infos.size()) {
        LOG(ERROR) << "onnxruntime session expects " << _m_input_infos.size() << " inputs, got "
                   << inputs.size();
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    try {
        const auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(inputs.size());
        for (const auto& named : inputs) {
            const auto info_iter = std::find_if(
                _m_input_infos.begin(), _m_input_infos.end(),
                [&named](const TensorInfo& info) { return info.name == named.name; });
            if (info_iter == _m_input_infos.end()) {
                LOG(ERROR) << "unknown onnxruntime input tensor: " << named.name;
                return StatusCode::MODEL_RUN_SESSION_FAILED;
            }
            if (info_iter->dtype != named.tensor.dtype) {
                LOG(ERROR) << "onnxruntime input '" << named.name << "' dtype mismatch, expected "
                           << info_iter->to_string() << ", got "
                           << dtype_to_string(named.tensor.dtype);
                return StatusCode::MODEL_RUN_SESSION_FAILED;
            }
            if (!named.tensor.shape_is_concrete()) {
                LOG(ERROR) << "onnxruntime input '" << named.name << "' has non concrete shape "
                           << shape_to_string(named.tensor.shape);
                return StatusCode::MODEL_RUN_SESSION_FAILED;
            }
            ort_inputs.push_back(Ort::Value::CreateTensor(
                memory_info, const_cast<uint8_t*>(named.tensor.buffer.data()),
                named.tensor.byte_size(), named.tensor.shape.data(), named.tensor.shape.size(),
                to_ort_dtype(named.tensor.dtype)));
        }

        auto ort_outputs = _m_session->Run(
            Ort::RunOptions{nullptr}, _m_input_name_ptrs.data(), ort_inputs.data(),
            ort_inputs.size(), _m_output_name_ptrs.data(), _m_output_name_ptrs.size());

        outputs.clear();
        outputs.reserve(ort_outputs.size());
        for (size_t idx = 0; idx < ort_outputs.size(); ++idx) {
            const auto& value = ort_outputs[idx];
            NamedTensor named;
            named.name = _m_output_infos[idx].name;
            const auto type_and_shape = value.GetTensorTypeAndShapeInfo();
            std::string dtype_err;
            if (!dtype_from_ort(type_and_shape.GetElementType(), &named.tensor.dtype,
                                &dtype_err)) {
                LOG(ERROR) << "onnxruntime output '" << named.name << "': " << dtype_err;
                return StatusCode::MODEL_RUN_SESSION_FAILED;
            }
            named.tensor.shape = type_and_shape.GetShape();
            const auto element_count = shape_volume(named.tensor.shape);
            if (element_count <= 0) {
                LOG(ERROR) << "onnxruntime output '" << named.name << "' is empty";
                return StatusCode::MODEL_EMPTY_OUTPUT;
            }
            const auto bytes = static_cast<size_t>(element_count) * dtype_size(named.tensor.dtype);
            named.tensor.buffer.resize(bytes);
            const void* src = ort_tensor_data(value, named.tensor.dtype);
            if (src == nullptr) {
                LOG(ERROR) << "onnxruntime output '" << named.name << "' data is null";
                return StatusCode::MODEL_RUN_SESSION_FAILED;
            }
            std::memcpy(named.tensor.buffer.data(), src, bytes);
            outputs.push_back(std::move(named));
        }
    } catch (const Ort::Exception& exception) {
        LOG(ERROR) << "onnxruntime run failed: " << exception.what();
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }
    return StatusCode::OK;
}

}  // namespace backend
}  // namespace models
}  // namespace jinq
