#ifndef MORTRED_MODELS_BACKEND_TENSOR_CONTRACT_H
#define MORTRED_MODELS_BACKEND_TENSOR_CONTRACT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "models/backend/tensor.h"

namespace jinq {
namespace models {
namespace backend {

struct TensorContract {
    DType dtype = DType::F32;
    size_t rank = 0;
    std::vector<int64_t> shape; // -1 means any positive dimension
};

inline const NamedTensor *find_output(const std::vector<NamedTensor> &outputs, const std::string &name) {
    const auto iter = std::find_if(outputs.begin(), outputs.end(), [&name](const NamedTensor &item) { return item.name == name; });
    return iter == outputs.end() ? nullptr : &*iter;
}

inline bool checked_element_count(const Tensor &tensor, size_t *byte_size, std::string *error) {
    size_t bytes = dtype_size(tensor.dtype);
    for (const int64_t dim : tensor.shape) {
        if (dim <= 0) {
            if (error != nullptr) {
                *error = "non-positive dimension " + std::to_string(dim);
            }
            return false;
        }
        const uint64_t dim_u = static_cast<uint64_t>(dim);
        if (bytes > std::numeric_limits<size_t>::max() / dim_u) {
            if (error != nullptr) {
                *error = "shape byte size overflows";
            }
            return false;
        }
        bytes *= static_cast<size_t>(dim_u);
    }
    *byte_size = bytes;
    return true;
}

inline bool validate_output_tensor(const NamedTensor &output, const TensorContract &contract, std::string *error) {
    if (output.name.empty()) {
        if (error != nullptr) {
            *error = "output tensor name is empty";
        }
        return false;
    }
    if (output.tensor.dtype != contract.dtype) {
        if (error != nullptr) {
            *error = "output '" + output.name + "' dtype is " + dtype_to_string(output.tensor.dtype) + ", expected " +
                     dtype_to_string(contract.dtype);
        }
        return false;
    }
    if (output.tensor.shape.size() != contract.rank) {
        if (error != nullptr) {
            *error = "output '" + output.name + "' rank is " + std::to_string(output.tensor.shape.size()) + ", expected " +
                     std::to_string(contract.rank);
        }
        return false;
    }
    if (contract.shape.size() != contract.rank) {
        if (error != nullptr) {
            *error = "internal contract rank does not match dimension count";
        }
        return false;
    }
    for (size_t idx = 0; idx < contract.rank; ++idx) {
        if (contract.shape[idx] >= 0 && output.tensor.shape[idx] != contract.shape[idx]) {
            if (error != nullptr) {
                *error = "output '" + output.name + "' dim[" + std::to_string(idx) + "] is " + std::to_string(output.tensor.shape[idx]) +
                         ", expected " + std::to_string(contract.shape[idx]);
            }
            return false;
        }
    }
    size_t expected_bytes = 0;
    if (!checked_element_count(output.tensor, &expected_bytes, error)) {
        if (error != nullptr && !error->empty()) {
            *error = "output '" + output.name + "': " + *error;
        }
        return false;
    }
    if (output.tensor.buffer.size() != expected_bytes) {
        if (error != nullptr) {
            *error = "output '" + output.name + "' buffer is " + std::to_string(output.tensor.buffer.size()) + " bytes, expected " +
                     std::to_string(expected_bytes);
        }
        return false;
    }
    return true;
}

inline bool get_f32_data(const Tensor &tensor, const float **data, std::string *error) {
    if (tensor.dtype != DType::F32) {
        if (error != nullptr) {
            *error = std::string("tensor dtype is ") + dtype_to_string(tensor.dtype) + ", expected f32";
        }
        return false;
    }
    size_t expected_bytes = 0;
    if (!checked_element_count(tensor, &expected_bytes, error)) {
        return false;
    }
    if (tensor.buffer.size() != expected_bytes) {
        if (error != nullptr) {
            *error = "tensor buffer is " + std::to_string(tensor.buffer.size()) + " bytes, expected " + std::to_string(expected_bytes);
        }
        return false;
    }
    *data = reinterpret_cast<const float *>(tensor.buffer.data());
    return true;
}

inline bool require_finite_f32(const float *data, size_t count, const std::string &output_name, std::string *error) {
    for (size_t idx = 0; idx < count; ++idx) {
        if (!std::isfinite(data[idx])) {
            if (error != nullptr) {
                *error = "output '" + output_name + "' has non-finite value at index " + std::to_string(idx);
            }
            return false;
        }
    }
    return true;
}

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_TENSOR_CONTRACT_H
