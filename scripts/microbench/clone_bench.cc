// Measures the cost of the full-image clone in ImagePipeline's constructor
// (models/backend/model_runtime.cpp:72) against the resize/convert chain that
// follows it, at the resolutions the server actually receives.
//
// Decision rule recorded before the numbers (see cowork rule: measure first):
//   clone <= 10% of the preprocess chain AND <= 3% of a typical request
//   => judge "not a defect", keep the clone and document the number;
//   otherwise drop it (shallow cv::Mat copy in the constructor).
//
// The chain below mirrors ImagePipeline::bgr_to_rgb().resize(net).to_float():
// each step allocates a new buffer, which is why the constructor's clone is a
// separate, avoidable copy of the same pixels.

#include <chrono>
#include <cstdio>
#include <functional>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

volatile const void *g_sink = nullptr;

double ms_per_op(int iters, const std::function<void()> &op) {
    op();  // warm up (page in, allocator warm)
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        op();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

} // namespace

int main() {
    struct Case {
        const char *name;
        int width;
        int height;
    };
    const Case cases[] = {{"640x640", 640, 640}, {"1080p", 1920, 1080}, {"4K", 3840, 2160}};
    const cv::Size net(640, 640);

    std::printf("%-9s %10s %10s %10s %8s\n", "source", "clone ms", "chain ms", "pixels MB", "clone %");
    for (const Case &c : cases) {
        const cv::Mat src(c.height, c.width, CV_8UC3, cv::Scalar(7, 8, 9));
        const int iters = (c.width * c.height > 4000000) ? 20 : 100;

        const double clone_ms = ms_per_op(iters, [&src]() {
            const cv::Mat copy = src.clone();
            g_sink = copy.data;
        });
        const double chain_ms = ms_per_op(iters, [&src, &net]() {
            cv::Mat rgb;
            cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
            cv::Mat resized;
            cv::resize(rgb, resized, net);
            cv::Mat floats;
            resized.convertTo(floats, CV_32FC3, 1.0 / 255.0);
            g_sink = floats.data;
        });
        const double pixels_mb = static_cast<double>(src.total() * src.elemSize()) / 1e6;
        std::printf("%-9s %10.3f %10.3f %10.1f %7.1f%%\n", c.name, clone_ms, chain_ms, pixels_mb,
                    100.0 * clone_ms / (clone_ms + chain_ms));
    }
    return 0;
}
