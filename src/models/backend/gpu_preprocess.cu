// CUDA kernel: fused planar YCbCr → letterbox → RGB → /255 → fp16 NCHW
// Reads jpeggpu's device-resident YCbCr planes and writes the TRT fp16
// input tensor in one pass — zero host round-trips.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__global__ void letterbox_ycbcr_fp16_kernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ cb_plane,
    const uint8_t* __restrict__ cr_plane,
    int src_w, int src_h,
    int cb_w, int cb_h,
    int unpad_w, int unpad_h,
    __half* __restrict__ output,   // fp16 NCHW [3][out_h][out_w]
    int out_w, int out_h,
    int pad_x, int pad_y,
    float pad_val,                  // 114.0f / 255.0f
    float inv_255)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (w >= out_w || h >= out_h) return;

    const int plane_size = out_h * out_w;
    __half* r_ptr = output + 0 * plane_size + (size_t)h * out_w + w;
    __half* g_ptr = output + 1 * plane_size + (size_t)h * out_w + w;
    __half* b_ptr = output + 2 * plane_size + (size_t)h * out_w + w;

    // letterbox inverse mapping: output (w,h) → source (sx,sy)
    const int lx = w - pad_x;
    const int ly = h - pad_y;

    if (lx < 0 || lx >= unpad_w || ly < 0 || ly >= unpad_h) {
        *r_ptr = __float2half(pad_val);
        *g_ptr = __float2half(pad_val);
        *b_ptr = __float2half(pad_val);
        return;
    }

    // nearest-neighbor resize from unpadded to source
    const int sx = (lx * src_w) / unpad_w;
    const int sy = (ly * src_h) / unpad_h;

    // sample Y (full res) and Cb/Cr (possibly half res, nearest-neighbor upsample)
    const float yv = (float)y_plane[sy * src_w + sx];
    const int cx = (sx * cb_w) / src_w;
    const int cy = (sy * cb_h) / src_h;
    const float cb = (float)cb_plane[cy * cb_w + cx] - 128.0f;
    const float cr = (float)cr_plane[cy * cb_w + cx] - 128.0f;

    // BT.601 full-range YCbCr → RGB
    const float r = fmaxf(0.0f, fminf(255.0f, yv + 1.402f * cr)) * inv_255;
    const float g = fmaxf(0.0f, fminf(255.0f, yv - 0.344136f * cb - 0.714136f * cr)) * inv_255;
    const float b = fmaxf(0.0f, fminf(255.0f, yv + 1.772f * cb)) * inv_255;

    *r_ptr = __float2half(r);
    *g_ptr = __float2half(g);
    *b_ptr = __float2half(b);
}

extern "C" cudaError_t launch_letterbox_ycbcr_fp16(
    const uint8_t* d_y, const uint8_t* d_cb, const uint8_t* d_cr,
    int src_w, int src_h, int cb_w, int cb_h,
    int unpad_w, int unpad_h,
    __half* d_out, int out_w, int out_h,
    int pad_x, int pad_y,
    cudaStream_t stream)
{
    const float pad_val = 114.0f / 255.0f;
    const float inv_255 = 1.0f / 255.0f;
    const dim3 block(256, 1, 1);
    const dim3 grid((out_w + block.x - 1) / block.x, out_h, 1);
    letterbox_ycbcr_fp16_kernel<<<grid, block, 0, stream>>>(
        d_y, d_cb, d_cr, src_w, src_h, cb_w, cb_h,
        unpad_w, unpad_h, d_out, out_w, out_h, pad_x, pad_y,
        pad_val, inv_255);
    return cudaGetLastError();
}
