// Baseline + candidate microbenchmark for src/models/backend/nchw_host.h (phase 0).
// v0 = current element-wise implementation (copied verbatim from the header).
// v1 = per-pixel channel-vector memcpy (the contiguous run in NHWC src is along C).
// v2 = cache-blocked (32x32 h-w tiles) variant of v1.
// Bit-exactness of v1/v2 vs v0 is asserted in-process; the accepted implementation
// will be verified again in mnn_layout_unittest during phase 4.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static bool permute_v0(const uint8_t *src, uint8_t *dst, int64_t n, int64_t h, int64_t w,
                       int64_t c, size_t es) {
    for (int64_t ni = 0; ni < n; ++ni)
        for (int64_t hi = 0; hi < h; ++hi)
            for (int64_t wi = 0; wi < w; ++wi)
                for (int64_t ci = 0; ci < c; ++ci) {
                    const size_t s = (((ni * h + hi) * w + wi) * c + ci) * es;
                    const size_t d = (((ni * c + ci) * h + hi) * w + wi) * es;
                    std::memcpy(dst + d, src + s, es);
                }
    return true;
}

static bool permute_v1(const uint8_t *src, uint8_t *dst, int64_t n, int64_t h, int64_t w,
                       int64_t c, size_t es) {
    // contiguous run in NHWC is the channel vector: one memcpy per (n,h,w,ci),
    // dst channel stride is H*W*es
    for (int64_t ni = 0; ni < n; ++ni)
        for (int64_t hi = 0; hi < h; ++hi)
            for (int64_t wi = 0; wi < w; ++wi) {
                const size_t s = (((ni * h + hi) * w + wi) * c) * es;
                const size_t d = ((ni * c * h + hi) * w + wi) * es;
                for (int64_t ci = 0; ci < c; ++ci)
                    std::memcpy(dst + d + ci * h * w * es, src + s + ci * es, es);
            }
    return true;
}

static bool permute_v2(const uint8_t *src, uint8_t *dst, int64_t n, int64_t h, int64_t w,
                       int64_t c, size_t es) {
    constexpr int64_t kTile = 32;
    for (int64_t ni = 0; ni < n; ++ni)
        for (int64_t ci = 0; ci < c; ++ci) {
            uint8_t *dst_plane = dst + ((ni * c + ci) * h * w) * es;
            for (int64_t h0 = 0; h0 < h; h0 += kTile)
                for (int64_t w0 = 0; w0 < w; w0 += kTile) {
                    const int64_t h1 = std::min(h0 + kTile, h);
                    const int64_t w1 = std::min(w0 + kTile, w);
                    for (int64_t hi = h0; hi < h1; ++hi) {
                        const uint8_t *src_row = src + ((ni * h + hi) * w) * c * es + ci * es;
                        uint8_t *dst_row = dst_plane + hi * w * es;
                        for (int64_t wi = w0; wi < w1; ++wi)
                            std::memcpy(dst_row + wi * es, src_row + wi * c * es, es);
                    }
                }
        }
    return true;
}

static bool permute_v3(const uint8_t *src, uint8_t *dst, int64_t n, int64_t h, int64_t w,
                       int64_t c, size_t es) {
    // dst-major: contiguous dst row writes, strided src gather, 4 pixels per pass
    for (int64_t ni = 0; ni < n; ++ni)
        for (int64_t ci = 0; ci < c; ++ci) {
            uint8_t *dst_plane = dst + ((ni * c + ci) * h * w) * es;
            for (int64_t hi = 0; hi < h; ++hi) {
                const uint8_t *__restrict src_row = src + ((ni * h + hi) * w) * c * es + ci * es;
                uint8_t *__restrict dst_row = dst_plane + hi * w * es;
                int64_t wi = 0;
                for (; wi + 4 <= w; wi += 4) {
                    std::memcpy(dst_row + (wi + 0) * es, src_row + (wi + 0) * c * es, es);
                    std::memcpy(dst_row + (wi + 1) * es, src_row + (wi + 1) * c * es, es);
                    std::memcpy(dst_row + (wi + 2) * es, src_row + (wi + 2) * c * es, es);
                    std::memcpy(dst_row + (wi + 3) * es, src_row + (wi + 3) * c * es, es);
                }
                for (; wi < w; ++wi)
                    std::memcpy(dst_row + wi * es, src_row + wi * c * es, es);
            }
        }
    return true;
}

static double bench(const std::function<bool(const uint8_t *, uint8_t *, int64_t, int64_t, int64_t,
                                             int64_t, size_t)> &fn,
                    const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int64_t n,
                    int64_t h, int64_t w, int64_t c, size_t es, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn(src.data(), dst.data(), n, h, w, c, es);
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

static void run(const char *name, int64_t n, int64_t h, int64_t w, int64_t c, size_t es) {
    std::vector<uint8_t> src(n * h * w * c * es, 0xAB), dst(src.size(), 0), ref(src.size(), 0);
    for (size_t i = 0; i < src.size(); i += es) src[i] = static_cast<uint8_t>(i);
    permute_v0(src.data(), ref.data(), n, h, w, c, es);
    const double t0 = bench(permute_v0, src, dst, n, h, w, c, es, 5);
    permute_v1(src.data(), dst.data(), n, h, w, c, es);
    if (std::memcmp(dst.data(), ref.data(), src.size()) != 0) {
        std::printf("%s v1 NOT BIT-EXACT\n", name);
        return;
    }
    const double t1 = bench(permute_v1, src, dst, n, h, w, c, es, 5);
    std::fill(dst.begin(), dst.end(), 0);
    permute_v2(src.data(), dst.data(), n, h, w, c, es);
    if (std::memcmp(dst.data(), ref.data(), src.size()) != 0) {
        std::printf("%s v2 NOT BIT-EXACT\n", name);
        return;
    }
    const double t2 = bench(permute_v2, src, dst, n, h, w, c, es, 5);
    std::fill(dst.begin(), dst.end(), 0);
    permute_v3(src.data(), dst.data(), n, h, w, c, es);
    if (std::memcmp(dst.data(), ref.data(), src.size()) != 0) {
        std::printf("%s v3 NOT BIT-EXACT\n", name);
        return;
    }
    const double t3 = bench(permute_v3, src, dst, n, h, w, c, es, 5);
    std::printf("%-28s v0=%7.2fms  v1=%7.2fms (%.1fx)  v2=%7.2fms (%.1fx)  v3=%7.2fms (%.1fx)\n",
                name, t0, t1, t0 / t1, t2, t0 / t2, t3, t0 / t3);
}

int main() {
    run("1x640x640x3 fp32 (yolo)", 1, 640, 640, 3, 4);
    run("1x224x224x3 fp32 (resnet)", 1, 224, 224, 3, 4);
    run("2x256x256x3 fp32", 2, 256, 256, 3, 4);
    run("1x512x512x4 fp16", 1, 512, 512, 4, 2);
    return 0;
}
