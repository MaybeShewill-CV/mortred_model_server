/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: backend/tensor.h
 * Date: 2026-08-20
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_TENSOR_H
#define MORTRED_MODELS_BACKEND_TENSOR_H

#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>

#include "glog/logging.h"

namespace jinq {
namespace models {
namespace backend {

/***
 * Dtype-erased host tensor. The buffer owns the data (no raw pointers),
 * the shape is always concrete at runtime (-1 only appears in TensorInfo).
 */
enum class DType {
    F32,
    I32,
    I64,
    U8,
};

inline size_t dtype_size(const DType& dtype) {
    switch (dtype) {
        case DType::F32:
            return sizeof(float);
        case DType::I32:
            return sizeof(int32_t);
        case DType::I64:
            return sizeof(int64_t);
        case DType::U8:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}

inline const char* dtype_to_string(const DType& dtype) {
    switch (dtype) {
        case DType::F32:
            return "f32";
        case DType::I32:
            return "i32";
        case DType::I64:
            return "i64";
        case DType::U8:
            return "u8";
        default:
            return "unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, const DType& dtype) {
    return os << dtype_to_string(dtype);
}

template<typename T>
inline DType dtype_of() {
    if constexpr (std::is_same<T, float>::value) {
        return DType::F32;
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return DType::I32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
        return DType::I64;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        return DType::U8;
    } else {
        static_assert(sizeof(T) == 0, "unsupported tensor element type, use f32/i32/i64/u8");
    }
}

/****
 * product of shape dims; -1 dims yield a negative result on purpose so that
 * callers treat the shape as non concrete
 */
inline int64_t shape_volume(const std::vector<int64_t>& shape) {
    int64_t volume = 1;
    for (const auto& dim : shape) {
        volume *= dim;
    }
    return volume;
}

inline bool shape_is_dynamic(const std::vector<int64_t>& shape) {
    for (const auto& dim : shape) {
        // -1 for onnx/tensorrt profiles, 0 for unset mnn dims
        if (dim <= 0) {
            return true;
        }
    }
    return false;
}

inline bool shape_equal(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
    return lhs == rhs;
}

inline std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        out += std::to_string(shape[idx]);
        if (idx + 1 < shape.size()) {
            out += ",";
        }
    }
    out += "]";
    return out;
}

struct Tensor {
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> buffer;

    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    /*** zero-initialized tensor of the given dtype and concrete shape */
    static Tensor make(const DType& dtype, const std::vector<int64_t>& shape) {
        Tensor tensor;
        tensor.dtype = dtype;
        tensor.shape = shape;
        const auto element_count = shape_volume(shape);
        CHECK_GT(element_count, 0) << "tensor shape must be concrete and non-empty: "
                                   << shape_to_string(shape);
        tensor.buffer.assign(static_cast<size_t>(element_count) * dtype_size(dtype), 0);
        return tensor;
    }

    template<typename T>
    static Tensor make(const std::vector<int64_t>& shape) {
        return make(dtype_of<T>(), shape);
    }

    /***
     * deep copy of a cv::Mat in HWC byte order (matching an nhwc layout);
     * only CV_8U and CV_32F mats are accepted, other depths are rejected
     */
    static Tensor from_mat(const cv::Mat& image, bool* ok = nullptr) {
        Tensor tensor;
        if (image.empty()) {
            LOG(ERROR) << "cannot build tensor from an empty cv::Mat";
            if (ok != nullptr) {
                *ok = false;
            }
            return tensor;
        }
        if (image.depth() == CV_8U) {
            tensor.dtype = DType::U8;
        } else if (image.depth() == CV_32F) {
            tensor.dtype = DType::F32;
        } else {
            LOG(ERROR) << "unsupported cv::Mat depth for tensor conversion: " << image.depth();
            if (ok != nullptr) {
                *ok = false;
            }
            return tensor;
        }
        tensor.shape = {1, image.rows, image.cols, image.channels()};
        const auto bytes = image.total() * image.elemSize();
        tensor.buffer.resize(bytes);
        if (image.isContinuous()) {
            std::memcpy(tensor.buffer.data(), image.data, bytes);
        } else {
            uint8_t* dst = tensor.buffer.data();
            for (int row = 0; row < image.rows; ++row) {
                const auto row_bytes = static_cast<size_t>(image.cols) * image.elemSize();
                std::memcpy(dst, image.ptr(row), row_bytes);
                dst += row_bytes;
            }
        }
        if (ok != nullptr) {
            *ok = true;
        }
        return tensor;
    }

    int64_t element_count() const {
        return shape_volume(shape);
    }

    size_t byte_size() const {
        return buffer.size();
    }

    bool shape_is_concrete() const {
        return !shape.empty() && !shape_is_dynamic(shape);
    }

    /*** typed view of the owned buffer; dtype mismatch is a hard error */
    template<typename T>
    T* data() {
        CHECK_EQ(dtype, dtype_of<T>()) << "tensor dtype is " << dtype_to_string(dtype);
        return reinterpret_cast<T*>(buffer.data());
    }

    template<typename T>
    const T* data() const {
        CHECK_EQ(dtype, dtype_of<T>()) << "tensor dtype is " << dtype_to_string(dtype);
        return reinterpret_cast<const T*>(buffer.data());
    }
};

struct NamedTensor {
    std::string name;
    Tensor tensor;
};

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_TENSOR_H
