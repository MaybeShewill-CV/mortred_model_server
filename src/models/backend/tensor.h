/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: tensor.h
* Date: 26-8-20
************************************************/

#ifndef MORTRED_MODELS_BACKEND_TENSOR_H
#define MORTRED_MODELS_BACKEND_TENSOR_H

#include <cstdint>
#include <limits>
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
    F16,
    I32,
    I64,
    U8,
};

inline size_t dtype_size(const DType& dtype) {
    switch (dtype) {
        case DType::F32:
            return sizeof(float);
        case DType::F16:
            return sizeof(uint16_t);
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
        case DType::F16:
            return "f16";
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

/***
 * Host memory layout of a Tensor. Rank-4 vision outputs from run() are NCHW.
 * Rank-1/2 scores are Linear. Rank-3 stays Unknown (YOLO rows vs HWC maps).
 * Unknown on a TensorContract means "do not check".
 */
enum class TensorLayout {
    Unknown,
    Nchw,
    Nhwc,
    Linear,
};

inline const char* tensor_layout_to_string(const TensorLayout& layout) {
    switch (layout) {
        case TensorLayout::Nchw:
            return "nchw";
        case TensorLayout::Nhwc:
            return "nhwc";
        case TensorLayout::Linear:
            return "linear";
        default:
            return "unknown";
    }
}

inline TensorLayout host_output_layout(const std::vector<int64_t>& shape) {
    if (shape.size() == 4) {
        // Channel-minor NHWC ([N,H,W,C] with C in 1..4 and H not a channel)
        // must not be tagged NCHW: MobileNet/ResNet/DenseNet/BiSeNet engines
        // are [1,224,224,3] / [1,H,W,3] and SessionIoValidator.nhwc() reads this tag.
        const int64_t d1 = shape[1];
        const int64_t d3 = shape[3];
        const bool d1_is_c = d1 > 0 && d1 <= 4;
        const bool d3_is_c = d3 > 0 && d3 <= 4;
        if (d3_is_c && !d1_is_c) {
            return TensorLayout::Nhwc;
        }
        return TensorLayout::Nchw;
    }
    if (shape.size() == 1 || shape.size() == 2) {
        return TensorLayout::Linear;
    }
    return TensorLayout::Unknown;
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
 * Product of shape dims.
 * - Any dim < 0 (dynamic marker): legacy multiply so the product is typically
 *   negative and callers treat the shape as non-concrete.
 * - Any dim == 0: returns 0.
 * - All dims > 0: returns the product, or -1 if the product would overflow
 *   int64_t (so size_t casts cannot turn a wrap into a huge allocation).
 */
inline int64_t shape_volume(const std::vector<int64_t>& shape) {
    for (const auto dim : shape) {
        if (dim < 0) {
            int64_t volume = 1;
            for (const auto d : shape) {
                volume *= d;
            }
            return volume;
        }
    }
    int64_t volume = 1;
    for (const auto dim : shape) {
        if (dim == 0) {
            return 0;
        }
        if (volume > std::numeric_limits<int64_t>::max() / dim) {
            return -1;
        }
        volume *= dim;
    }
    return volume;
}

/*** Byte size for a concrete shape; false on non-positive dims or size_t overflow. */
inline bool checked_shape_nbytes(const std::vector<int64_t>& shape, DType dtype, size_t* nbytes) {
    size_t bytes = dtype_size(dtype);
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            return false;
        }
        const uint64_t dim_u = static_cast<uint64_t>(dim);
        if (bytes > std::numeric_limits<size_t>::max() / dim_u) {
            return false;
        }
        bytes *= static_cast<size_t>(dim_u);
    }
    if (nbytes != nullptr) {
        *nbytes = bytes;
    }
    return true;
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
    TensorLayout layout = TensorLayout::Unknown;
    // GPU zero-copy: if non-null, data lives in GPU memory at this address
    // and `buffer` is empty. Sessions skip H2D and use this pointer directly.
    void* device_data = nullptr;
    // cudaEvent_t recorded on the producing stream after the last write to
    // device_data; consumer sessions MUST cudaStreamWaitEvent on it before
    // reading — producer and consumer run on different streams.
    void* device_ready_event = nullptr;

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
        tensor.layout = host_output_layout(shape);
        size_t nbytes = 0;
        CHECK(checked_shape_nbytes(shape, dtype, &nbytes))
            << "tensor shape must be concrete, non-empty, and fit in size_t: "
            << shape_to_string(shape);
        tensor.buffer.assign(nbytes, 0);
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
        tensor.layout = TensorLayout::Nhwc;
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
