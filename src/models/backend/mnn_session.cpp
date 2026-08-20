/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: backend/mnn_session.cpp
 * Date: 2026-08-20
 ************************************************/

#include "models/backend/mnn_session.h"

#include <algorithm>
#include <cstring>

#include "glog/logging.h"

#include "common/file_path_util.h"

namespace jinq {
namespace models {
namespace backend {

using jinq::common::StatusCode;

namespace {

bool dtype_from_mnn(const MNN::Tensor* tensor, DType* out, std::string* err) {
    const auto type = tensor->getType();
    if (type.code == halide_type_float && type.bits == 32) {
        *out = DType::F32;
    } else if (type.code == halide_type_int && type.bits == 32) {
        *out = DType::I32;
    } else if (type.code == halide_type_int && type.bits == 64) {
        *out = DType::I64;
    } else if (type.code == halide_type_uint && type.bits == 8) {
        *out = DType::U8;
    } else {
        if (err != nullptr) {
            *err = "unsupported mnn tensor dtype (code=" + std::to_string(static_cast<int>(type.code))
                   + ", bits=" + std::to_string(static_cast<int>(type.bits)) + ")";
        }
        return false;
    }
    return true;
}

std::vector<int> to_mnn_dims(const std::vector<int64_t>& shape) {
    std::vector<int> dims;
    dims.reserve(shape.size());
    for (const auto& dim : shape) {
        dims.push_back(static_cast<int>(dim));
    }
    return dims;
}

/*** mnn stores dims canonically (nchw); host layouts follow the dim type */
std::vector<int64_t> to_host_shape(const std::vector<int>& dims,
                                   const MNN::Tensor::DimensionType& dim_type) {
    std::vector<int64_t> shape;
    shape.reserve(dims.size());
    for (const auto& dim : dims) {
        shape.push_back(dim);
    }
    if (dim_type == MNN::Tensor::DimensionType::TENSORFLOW && shape.size() == 4) {
        return {shape[0], shape[2], shape[3], shape[1]};
    }
    return shape;
}

std::vector<int> to_internal_dims(const std::vector<int64_t>& host_shape,
                                  const MNN::Tensor::DimensionType& dim_type) {
    if (dim_type == MNN::Tensor::DimensionType::TENSORFLOW && host_shape.size() == 4) {
        return {static_cast<int>(host_shape[0]), static_cast<int>(host_shape[3]),
                static_cast<int>(host_shape[1]), static_cast<int>(host_shape[2])};
    }
    return to_mnn_dims(host_shape);
}

}  // namespace

MnnSession::~MnnSession() {
    if (_m_interpreter == nullptr) {
        return;
    }
    if (_m_session != nullptr) {
        _m_interpreter->releaseSession(_m_session);
        _m_session = nullptr;
    }
    _m_interpreter->releaseModel();
}

StatusCode MnnSession::init(const BackendConfig& config, std::string* err) {
    if (err != nullptr) {
        err->clear();
    }
    if (!jinq::common::FilePathUtil::is_file_exist(config.model_file_path)) {
        if (err != nullptr) {
            *err = "mnn model file not exist: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }

    _m_interpreter.reset(MNN::Interpreter::createFromFile(config.model_file_path.c_str()));
    if (_m_interpreter == nullptr) {
        if (err != nullptr) {
            *err = "create mnn interpreter failed: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }

    MNN::ScheduleConfig schedule_config;
    if (config.use_cuda()) {
        schedule_config.type = MNN_FORWARD_CUDA;
    } else {
        schedule_config.type = MNN_FORWARD_CPU;
    }
    schedule_config.numThread = config.threads;

    MNN::BackendConfig backend_config;
    backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(config.precision_mode);
    backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(config.power_mode);
    schedule_config.backendConfig = &backend_config;

    _m_session = _m_interpreter->createSession(schedule_config);
    if (_m_session == nullptr) {
        if (err != nullptr) {
            *err = "create mnn session failed: " + config.model_file_path;
        }
        _m_interpreter->releaseModel();
        _m_interpreter.reset();
        return StatusCode::MODEL_INIT_FAILED;
    }

    const auto all_inputs = _m_interpreter->getSessionInputAll(_m_session);
    const auto all_outputs = _m_interpreter->getSessionOutputAll(_m_session);
    std::map<std::string, MNN::Tensor::DimensionType> input_dim_types;
    for (const auto& item : all_inputs) {
        if (item.second == nullptr) {
            continue;
        }
        if (!config.input_names.empty() &&
            std::find(config.input_names.begin(), config.input_names.end(), item.first) ==
                config.input_names.end()) {
            continue;
        }
        auto dim_type = item.second->getDimensionType();
        if (config.input_layout == "nhwc") {
            dim_type = MNN::Tensor::DimensionType::TENSORFLOW;
        } else if (config.input_layout == "nchw") {
            dim_type = MNN::Tensor::DimensionType::CAFFE;
        }
        input_dim_types[item.first] = dim_type;
        _m_input_tensors[item.first] = item.second;
    }
    for (const auto& item : all_outputs) {
        if (item.second == nullptr) {
            continue;
        }
        if (!config.output_names.empty() &&
            std::find(config.output_names.begin(), config.output_names.end(), item.first) ==
                config.output_names.end()) {
            continue;
        }
        _m_output_tensors[item.first] = item.second;
    }
    if (_m_input_tensors.empty() || _m_output_tensors.empty()) {
        if (err != nullptr) {
            *err = "mnn model exposes no io tensors: " + config.model_file_path;
        }
        return StatusCode::MODEL_INIT_FAILED;
    }
    for (const auto& name : config.input_names) {
        if (_m_input_tensors.find(name) == _m_input_tensors.end()) {
            if (err != nullptr) {
                *err = "configured mnn input tensor not found: " + name;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
    }
    for (const auto& name : config.output_names) {
        if (_m_output_tensors.find(name) == _m_output_tensors.end()) {
            if (err != nullptr) {
                *err = "configured mnn output tensor not found: " + name;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
    }

    // build infos last: dtypes must be valid, shapes are reported in the host
    // layout used for copies (nhwc for TENSORFLOW dim type tensors)
    for (const auto& item : _m_input_tensors) {
        TensorInfo info;
        info.name = item.first;
        std::string parse_err;
        if (!dtype_from_mnn(item.second, &info.dtype, &parse_err)) {
            if (err != nullptr) {
                *err = "mnn input '" + item.first + "': " + parse_err;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
        info.shape = to_host_shape(item.second->shape(), input_dim_types.at(item.first));
        info.dynamic = shape_is_dynamic(info.shape);
        _m_input_dim_types[item.first] = input_dim_types.at(item.first);
        _m_input_infos.push_back(std::move(info));
    }
    for (const auto& item : _m_output_tensors) {
        TensorInfo info;
        info.name = item.first;
        std::string parse_err;
        if (!dtype_from_mnn(item.second, &info.dtype, &parse_err)) {
            if (err != nullptr) {
                *err = "mnn output '" + item.first + "': " + parse_err;
            }
            return StatusCode::MODEL_INIT_FAILED;
        }
        info.shape = to_host_shape(item.second->shape(), item.second->getDimensionType());
        info.dynamic = shape_is_dynamic(info.shape);
        _m_output_infos.push_back(std::move(info));
    }

    _m_model_file_path = config.model_file_path;
    for (const auto& info : _m_input_infos) {
        LOG(INFO) << "mnn session input: " << info.to_string();
    }
    for (const auto& info : _m_output_infos) {
        LOG(INFO) << "mnn session output: " << info.to_string();
    }
    return StatusCode::OK;
}

StatusCode MnnSession::refresh_io_tensors() {
    for (auto& item : _m_input_tensors) {
        item.second = _m_interpreter->getSessionInput(_m_session, item.first.c_str());
        if (item.second == nullptr) {
            LOG(ERROR) << "mnn input tensor lost after resize: " << item.first;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
    }
    for (auto& item : _m_output_tensors) {
        item.second = _m_interpreter->getSessionOutput(_m_session, item.first.c_str());
        if (item.second == nullptr) {
            LOG(ERROR) << "mnn output tensor lost after resize: " << item.first;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
    }
    return StatusCode::OK;
}

StatusCode MnnSession::run(const std::vector<NamedTensor>& inputs,
                           std::vector<NamedTensor>& outputs) {
    if (_m_interpreter == nullptr || _m_session == nullptr) {
        LOG(ERROR) << "mnn session is not initialized";
        return StatusCode::MODEL_INIT_FAILED;
    }
    if (inputs.size() != _m_input_tensors.size()) {
        LOG(ERROR) << "mnn session expects " << _m_input_tensors.size() << " inputs, got "
                   << inputs.size();
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    bool need_resize = false;
    for (const auto& named : inputs) {
        const auto iter = _m_input_tensors.find(named.name);
        if (iter == _m_input_tensors.end() || iter->second == nullptr) {
            LOG(ERROR) << "unknown mnn input tensor: " << named.name;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        auto* mnn_tensor = iter->second;
        const auto dtype_iter = std::find_if(
            _m_input_infos.begin(), _m_input_infos.end(),
            [&named](const TensorInfo& info) { return info.name == named.name; });
        if (dtype_iter == _m_input_infos.end() || dtype_iter->dtype != named.tensor.dtype) {
            LOG(ERROR) << "mnn input '" << named.name << "' dtype mismatch, expected "
                       << (dtype_iter == _m_input_infos.end() ? std::string("unknown")
                                                              : dtype_iter->to_string())
                       << ", got " << dtype_to_string(named.tensor.dtype);
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        if (!named.tensor.shape_is_concrete()) {
            LOG(ERROR) << "mnn input '" << named.name << "' has non concrete shape "
                       << shape_to_string(named.tensor.shape);
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        const auto dim_type = _m_input_dim_types.at(named.name);
        const auto current_host_shape = to_host_shape(mnn_tensor->shape(), dim_type);
        if (!shape_equal(current_host_shape, named.tensor.shape)) {
            _m_interpreter->resizeTensor(mnn_tensor, to_internal_dims(named.tensor.shape, dim_type));
            need_resize = true;
        }
    }
    if (need_resize) {
        _m_interpreter->resizeSession(_m_session);
        const auto refresh_status = refresh_io_tensors();
        if (refresh_status != StatusCode::OK) {
            return refresh_status;
        }
    }

    for (const auto& named : inputs) {
        auto* mnn_tensor = _m_input_tensors.at(named.name);
        const auto dim_type = _m_input_dim_types.at(named.name);
        MNN::Tensor host_tensor(mnn_tensor, dim_type);
        const auto host_bytes = static_cast<size_t>(host_tensor.size());
        if (host_bytes != named.tensor.byte_size()) {
            LOG(ERROR) << "mnn input '" << named.name << "' byte size mismatch, expected "
                       << host_bytes << ", got " << named.tensor.byte_size();
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        std::memcpy(host_tensor.host<void>(), named.tensor.buffer.data(), host_bytes);
        mnn_tensor->copyFromHostTensor(&host_tensor);
    }

    const auto error_code = _m_interpreter->runSession(_m_session);
    if (error_code != MNN::NO_ERROR) {
        LOG(ERROR) << "run mnn session failed, mnn error code: " << static_cast<int>(error_code);
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    outputs.clear();
    outputs.reserve(_m_output_tensors.size());
    for (const auto& info : _m_output_infos) {
        auto* mnn_tensor = _m_output_tensors.at(info.name);
        if (mnn_tensor == nullptr) {
            LOG(ERROR) << "mnn output tensor missing: " << info.name;
            return StatusCode::MODEL_RUN_SESSION_FAILED;
        }
        MNN::Tensor host_tensor(mnn_tensor, mnn_tensor->getDimensionType());
        mnn_tensor->copyToHostTensor(&host_tensor);

        NamedTensor named;
        named.name = info.name;
        named.tensor.dtype = info.dtype;
        named.tensor.shape = to_host_shape(host_tensor.shape(), mnn_tensor->getDimensionType());
        const auto bytes = static_cast<size_t>(host_tensor.size());
        named.tensor.buffer.resize(bytes);
        std::memcpy(named.tensor.buffer.data(), host_tensor.host<void>(), bytes);
        outputs.push_back(std::move(named));
    }
    return StatusCode::OK;
}

}  // namespace backend
}  // namespace models
}  // namespace jinq
