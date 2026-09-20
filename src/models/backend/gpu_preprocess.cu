// Parameterized CUDA kernel: planar YCbCr → resize → rotate → color convert
// → normalize → fp16/f32 NCHW/NHWC, all on device in one pass.
// Reads jpeggpu's device-resident YCbCr planes and writes the TRT input
// tensor — zero host round-trips.
//
// All geometric/photometric parameters come from GpuPreprocessDescriptor
// (value-passed to the kernel, no pointers to host memory).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ── Resize types (match GpuPreprocessDescriptor::Resize) ──
#define RESIZE_LETTERBOX             0
#define RESIZE_CENTER_CROP            1
#define RESIZE_DIRECT_RESIZE          2
#define RESIZE_KEEP_RATIO_PAD_ZERO    3
#define RESIZE_KEEP_RATIO_PAD_CENTER  4
#define RESIZE_ALIGN_TO_MULTIPLE      5
#define RESIZE_NONE                   6

// ── Color orders (match GpuPreprocessDescriptor::Color) ──
#define COLOR_RGB  0
#define COLOR_BGR  1
#define COLOR_GRAY 2

// ── Rotations (match GpuPreprocessDescriptor::Rotation) ──
#define ROT_NONE 0
#define ROT_90   1
#define ROT_180  2
#define ROT_270  3

// ── Helper: write one pixel (defined BEFORE the kernel that uses it) ──
template <bool IS_FP16, bool IS_NHWC>
__device__ __forceinline__ void write_pixel(
    void* output, int w, int h, int out_w, int out_h,
    float ch0, float ch1, float ch2)
{
    if constexpr (IS_FP16) {
        __half* out = (__half*)output;
        if constexpr (IS_NHWC) {
            const size_t idx = ((size_t)h * out_w + w) * 3;
            out[idx + 0] = __float2half(ch0);
            out[idx + 1] = __float2half(ch1);
            out[idx + 2] = __float2half(ch2);
        } else {
            const size_t plane = (size_t)out_h * out_w;
            out[0 * plane + (size_t)h * out_w + w] = __float2half(ch0);
            out[1 * plane + (size_t)h * out_w + w] = __float2half(ch1);
            out[2 * plane + (size_t)h * out_w + w] = __float2half(ch2);
        }
    } else {
        float* out = (float*)output;
        if constexpr (IS_NHWC) {
            const size_t idx = ((size_t)h * out_w + w) * 3;
            out[idx + 0] = ch0;
            out[idx + 1] = ch1;
            out[idx + 2] = ch2;
        } else {
            const size_t plane = (size_t)out_h * out_w;
            out[0 * plane + (size_t)h * out_w + w] = ch0;
            out[1 * plane + (size_t)h * out_w + w] = ch1;
            out[2 * plane + (size_t)h * out_w + w] = ch2;
        }
    }
}

__device__ __forceinline__ float clamp_255(float v) {
    return fmaxf(0.0f, fminf(255.0f, v));
}

/*** Core kernel: one thread per output pixel.
 * Output is passed as void* and cast internally by the template parameter.
 * All descriptor fields are scalar kernel params (GPU-friendly). */
template <bool IS_FP16, bool IS_NHWC>
__global__ void preprocess_ycbcr_kernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ cb_plane,
    const uint8_t* __restrict__ cr_plane,
    int src_w, int src_h,
    int cb_w, int cb_h,
    void* output,                    // cast to __half* or float* by template
    void* gray_output,               // optional secondary grayscale (null = skip)
    int out_w, int out_h,
    // descriptor params (scalars)
    int resize_type,
    int color_order,
    int rotation,
    float norm_scale,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2,
    float pad_val,
    // geometry (pre-computed on host)
    int unpad_w, int unpad_h,
    int pad_x, int pad_y,
    int crop_x, int crop_y)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (w >= out_w || h >= out_h) return;

    // ── Step 1: geometric mapping (output pixel → source pixel) ──
    int sx = -1, sy = -1;
    bool is_pad = false;

    switch (resize_type) {
    case RESIZE_LETTERBOX: {
        const int lx = w - pad_x;
        const int ly = h - pad_y;
        if (lx < 0 || lx >= unpad_w || ly < 0 || ly >= unpad_h) {
            is_pad = true;
        } else {
            sx = (lx * src_w) / unpad_w;
            sy = (ly * src_h) / unpad_h;
        }
        break;
    }
    case RESIZE_CENTER_CROP: {
        sx = ((w + crop_x) * src_w) / (out_w + 2 * crop_x);
        sy = ((h + crop_y) * src_h) / (out_h + 2 * crop_y);
        break;
    }
    case RESIZE_DIRECT_RESIZE: {
        sx = (w * src_w) / out_w;
        sy = (h * src_h) / out_h;
        break;
    }
    case RESIZE_KEEP_RATIO_PAD_ZERO:
    case RESIZE_KEEP_RATIO_PAD_CENTER: {
        const int lx = w - pad_x;
        const int ly = h - pad_y;
        if (lx < 0 || lx >= unpad_w || ly < 0 || ly >= unpad_h) {
            is_pad = true;
            pad_val = 0.0f;  // both variants pad with zero
        } else {
            sx = (lx * src_w) / unpad_w;
            sy = (ly * src_h) / unpad_h;
        }
        break;
    }
    case RESIZE_ALIGN_TO_MULTIPLE:
    case RESIZE_NONE: {
        sx = (w * src_w) / out_w;
        sy = (h * src_h) / out_h;
        break;
    }
    default:
        return;
    }

    // ── Apply rotation to source coordinates ──
    switch (rotation) {
    case ROT_90:   { int t = sx; sx = sy;           sy = src_w - 1 - t; break; }
    case ROT_180:  { sx = src_w - 1 - sx; sy = src_h - 1 - sy; break; }
    case ROT_270:  { int t = sx; sx = src_h - 1 - sy; sy = t;            break; }
    default: break;
    }

    // ── Write padding if outside image area ──
    if (is_pad || sx < 0 || sx >= src_w || sy < 0 || sy >= src_h) {
        const float pv = (pad_val * norm_scale - mean0) / std0;
        write_pixel<IS_FP16, IS_NHWC>(output, w, h, out_w, out_h, pv, pv, pv);
        return;
    }

    // ── Step 2: sample YCbCr (nearest-neighbor chroma upsampling) ──
    const float yv = (float)y_plane[sy * src_w + sx];
    const int cx = (sx * cb_w) / src_w;
    const int cy = (sy * cb_h) / src_h;
    const float cb = (float)cb_plane[cy * cb_w + cx] - 128.0f;
    const float cr = (float)cr_plane[cy * cb_w + cx] - 128.0f;

    // ── Step 3: YCbCr → RGB (BT.601 full-range) ──
    float r = clamp_255(yv + 1.402f * cr);
    float g = clamp_255(yv - 0.344136f * cb - 0.714136f * cr);
    float b = clamp_255(yv + 1.772f * cb);

    // ── Step 4: normalize: out = (in * scale - mean) / std ──
    r = (r * norm_scale - mean0) / std0;
    g = (g * norm_scale - mean1) / std1;
    b = (b * norm_scale - mean2) / std2;

    // ── Step 5: apply color order and write ──
    if (color_order == COLOR_BGR) {
        write_pixel<IS_FP16, IS_NHWC>(output, w, h, out_w, out_h, b, g, r);
    } else if (color_order == COLOR_GRAY) {
        const float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        write_pixel<IS_FP16, IS_NHWC>(output, w, h, out_w, out_h, gray, gray, gray);
    } else { // RGB
        write_pixel<IS_FP16, IS_NHWC>(output, w, h, out_w, out_h, r, g, b);
    }

    // ── Step 6: optional grayscale secondary output ──
    if (gray_output != nullptr) {
        const float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        if constexpr (IS_FP16) {
            ((__half*)gray_output)[(size_t)h * out_w + w] = __float2half(gray);
        } else {
            ((float*)gray_output)[(size_t)h * out_w + w] = gray;
        }
    }
}

/*** C launcher: dispatches to the correct template instantiation. */
extern "C" cudaError_t launch_preprocess(
    const uint8_t* d_y, const uint8_t* d_cb, const uint8_t* d_cr,
    int src_w, int src_h, int cb_w, int cb_h,
    void* d_out, void* d_gray_out,
    int out_w, int out_h,
    int resize_type, int color_order, int rotation,
    float norm_scale,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2,
    float pad_val,
    int output_is_fp16,
    int output_is_nhwc,
    int unpad_w, int unpad_h, int pad_x, int pad_y,
    int crop_x, int crop_y,
    cudaStream_t stream)
{
    const dim3 block(256, 1, 1);
    const dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);

    if (output_is_fp16 && !output_is_nhwc) {
        preprocess_ycbcr_kernel<true, false><<<grid, block, 0, stream>>>(
            d_y, d_cb, d_cr, src_w, src_h, cb_w, cb_h,
            d_out, d_gray_out, out_w, out_h,
            resize_type, color_order, rotation,
            norm_scale, mean0, mean1, mean2, std0, std1, std2,
            pad_val, unpad_w, unpad_h, pad_x, pad_y, crop_x, crop_y);
    } else if (output_is_fp16 && output_is_nhwc) {
        preprocess_ycbcr_kernel<true, true><<<grid, block, 0, stream>>>(
            d_y, d_cb, d_cr, src_w, src_h, cb_w, cb_h,
            d_out, d_gray_out, out_w, out_h,
            resize_type, color_order, rotation,
            norm_scale, mean0, mean1, mean2, std0, std1, std2,
            pad_val, unpad_w, unpad_h, pad_x, pad_y, crop_x, crop_y);
    } else if (!output_is_fp16 && !output_is_nhwc) {
        preprocess_ycbcr_kernel<false, false><<<grid, block, 0, stream>>>(
            d_y, d_cb, d_cr, src_w, src_h, cb_w, cb_h,
            d_out, d_gray_out, out_w, out_h,
            resize_type, color_order, rotation,
            norm_scale, mean0, mean1, mean2, std0, std1, std2,
            pad_val, unpad_w, unpad_h, pad_x, pad_y, crop_x, crop_y);
    } else {
        preprocess_ycbcr_kernel<false, true><<<grid, block, 0, stream>>>(
            d_y, d_cb, d_cr, src_w, src_h, cb_w, cb_h,
            d_out, d_gray_out, out_w, out_h,
            resize_type, color_order, rotation,
            norm_scale, mean0, mean1, mean2, std0, std1, std2,
            pad_val, unpad_w, unpad_h, pad_x, pad_y, crop_x, crop_y);
    }
    return cudaGetLastError();
}
