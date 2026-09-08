/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: nchw_host.h
 * Date: 26-9-8
 ************************************************/

#ifndef MORTRED_MODELS_BACKEND_NCHW_HOST_H
#define MORTRED_MODELS_BACKEND_NCHW_HOST_H

#include <cstdint>
#include <cstring>
#include <vector>

namespace jinq {
namespace models {
namespace backend {

/*** remap a rank-4 NHWC shape [N,H,W,C] to logical NCHW [N,C,H,W]; other ranks unchanged */
inline std::vector<int64_t> nchw_shape_from_nhwc4(const std::vector<int64_t>& shape) {
    if (shape.size() != 4) {
        return shape;
    }
    return {shape[0], shape[3], shape[1], shape[2]};
}

/*** permute NHWC bytes [N,H,W,C] into NCHW [N,C,H,W]; src and dst must not overlap */
inline bool permute_nhwc_to_nchw_bytes(const uint8_t* src, uint8_t* dst, int64_t n, int64_t h,
                                       int64_t w, int64_t c, size_t elem_size) {
    if (src == nullptr || dst == nullptr || src == dst || elem_size == 0) {
        return false;
    }
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0) {
        return false;
    }
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t hi = 0; hi < h; ++hi) {
            for (int64_t wi = 0; wi < w; ++wi) {
                for (int64_t ci = 0; ci < c; ++ci) {
                    const size_t src_idx = static_cast<size_t>(
                        (((ni * h + hi) * w + wi) * c + ci) * static_cast<int64_t>(elem_size));
                    const size_t dst_idx = static_cast<size_t>(
                        (((ni * c + ci) * h + hi) * w + wi) * static_cast<int64_t>(elem_size));
                    std::memcpy(dst + dst_idx, src + src_idx, elem_size);
                }
            }
        }
    }
    return true;
}

}  // namespace backend
}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_NCHW_HOST_H
