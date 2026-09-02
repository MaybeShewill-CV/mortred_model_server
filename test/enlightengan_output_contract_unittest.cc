#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"
#include "models/enhancement/enlightengan.h"
#include "models/io/enhancement.h"

using jinq::common::StatusCode;
using jinq::models::backend::InferenceContext;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using Output = jinq::models::io_define::enhancement::std_enhancement_output;

namespace {

template <typename MODEL> class CallableEnlightenGan : public MODEL {
  public:
    using MODEL::_m_input_dynamic;
    using MODEL::_m_input_size_host;
    using MODEL::postprocess;
    using MODEL::preprocess;
};

using TestModel = CallableEnlightenGan<jinq::models::enhancement::EnlightenGan<
    jinq::models::io_define::common_io::mat_input, Output>>;

// normalized [-1, 1] value of a 8-bit channel value after the model's
// x/255 -> (x - 0.5) / 0.5 chain
constexpr float normalized(uint8_t value) {
    return (static_cast<float>(value) / 255.0f - 0.5f) / 0.5f;
}

NamedTensor f32_output(const std::vector<int64_t> &shape, float fill) {
    NamedTensor tensor;
    tensor.name = "output";
    tensor.tensor = Tensor::make<float>(shape);
    auto *data = tensor.tensor.data<float>();
    for (int64_t idx = 0; idx < tensor.tensor.element_count(); ++idx) {
        data[idx] = fill;
    }
    return tensor;
}

} // namespace

TEST(EnlightenGanPreprocess, ResizePathKeepsRgbChannelOrder) {
    // regression: the old code converted BGR->RGB first and then resized the
    // RAW input over the converted buffer, silently feeding BGR to the
    // network whenever the request needed a resize (and re-introducing the
    // 4th channel for RGBA requests). The conversion must run after resize.
    TestModel model;
    model._m_input_size_host = cv::Size(256, 256); // fixed-shape session
    model._m_input_dynamic = false;

    // solid pure-red BGR image that MUST be resized (300x400 -> 256x256)
    const cv::Mat red(300, 400, CV_8UC3, cv::Scalar(0, 0, 255));
    const auto tensors = model.preprocess(red);
    ASSERT_EQ(tensors.size(), 2u);
    EXPECT_EQ(tensors[0].name, "input_src");
    EXPECT_EQ(tensors[1].name, "input_gray");
    ASSERT_EQ(tensors[0].tensor.shape, (std::vector<int64_t>{1, 3, 256, 256}));
    ASSERT_EQ(tensors[1].tensor.shape, (std::vector<int64_t>{1, 1, 256, 256}));

    // pure red in RGB order: R plane = +1, G/B planes = -1 (a BGR leak would
    // put the red value into plane 2 instead)
    const auto *src = tensors[0].tensor.data<float>();
    const size_t plane = 256u * 256u;
    for (const auto [row, col] : {std::pair<int, int>{0, 0}, {128, 128}, {255, 255}}) {
        const size_t idx = static_cast<size_t>(row) * 256 + col;
        EXPECT_NEAR(src[idx], normalized(255), 1e-4f) << "R plane at " << row << "," << col;
        EXPECT_NEAR(src[plane + idx], normalized(0), 1e-4f) << "G plane at " << row << "," << col;
        EXPECT_NEAR(src[2 * plane + idx], normalized(0), 1e-4f) << "B plane at " << row << "," << col;
    }
    // the gray map of pure red: 1 - 0.299 * (r + 1) with r = +1
    EXPECT_NEAR(tensors[1].tensor.data<float>()[0], 1.0f - 0.299f, 1e-4f);
}

TEST(EnlightenGanPreprocess, NoResizePathMatchesResizePath) {
    // an input that already has the network size must produce the exact same
    // normalized tensor as the resized one (channel order must not depend on
    // whether the resize branch ran)
    TestModel model;
    model._m_input_size_host = cv::Size(256, 256);
    model._m_input_dynamic = false;

    const cv::Mat red(256, 256, CV_8UC3, cv::Scalar(0, 0, 255));
    const auto tensors = model.preprocess(red);
    ASSERT_EQ(tensors.size(), 2u);
    ASSERT_EQ(tensors[0].tensor.shape, (std::vector<int64_t>{1, 3, 256, 256}));

    const auto *src = tensors[0].tensor.data<float>();
    EXPECT_NEAR(src[0], normalized(255), 1e-4f);
    EXPECT_NEAR(src[256u * 256u], normalized(0), 1e-4f);
    EXPECT_NEAR(src[2 * 256u * 256u], normalized(0), 1e-4f);
}

TEST(EnlightenGanPreprocess, RgbaResizeNoLongerFails) {
    // regression: resizing a 4-channel request rebuilt the destination as
    // CV_8UC4 (resize copies the source type), then the normalization split 4
    // planes and convert_to_chw_vec rejected CV_32FC4 -> empty preprocess.
    // Resize-then-convert keeps the alpha out of the network tensor.
    TestModel model;
    model._m_input_size_host = cv::Size(256, 256);
    model._m_input_dynamic = false;

    // BGRA: pure red with a semi-transparent alpha (0, 0, 255, 128)
    const cv::Mat bgra(300, 400, CV_8UC4, cv::Scalar(0, 0, 255, 128));
    const auto tensors = model.preprocess(bgra);
    ASSERT_EQ(tensors.size(), 2u);
    ASSERT_EQ(tensors[0].tensor.shape, (std::vector<int64_t>{1, 3, 256, 256}));
    ASSERT_EQ(tensors[1].tensor.shape, (std::vector<int64_t>{1, 1, 256, 256}));

    const auto *src = tensors[0].tensor.data<float>();
    const size_t plane = 256u * 256u;
    // alpha (128) must NOT leak into the color planes: R=+1, G/B=-1
    EXPECT_NEAR(src[0], normalized(255), 1e-4f);
    EXPECT_NEAR(src[plane], normalized(0), 1e-4f);
    EXPECT_NEAR(src[2 * plane], normalized(0), 1e-4f);
}

TEST(EnlightenGanPostprocess, RestoresSourceAlphaFromContext) {
    // the alpha plane must travel with the REQUEST (context.source_image),
    // not in a model member: batch items interleave pre/postprocess calls on
    // one shared instance, so member state would cross-contaminate items
    TestModel model;
    model._m_input_size_host = cv::Size(256, 256);
    model._m_input_dynamic = false;

    // 300x400 BGRA with a per-pixel alpha pattern
    cv::Mat bgra(300, 400, CV_8UC4);
    for (int row = 0; row < 300; ++row) {
        for (int col = 0; col < 400; ++col) {
            bgra.at<cv::Vec4b>(row, col) = cv::Vec4b(0, 0, 255, static_cast<uchar>((row * 7 + col * 13) % 256));
        }
    }
    // the model output for a mid-gray enhancement (f32 0.0 -> uchar 127)
    InferenceContext ctx;
    ctx.source_size = cv::Size(400, 300);
    ctx.network_size = cv::Size(256, 256);
    ctx.source_image = bgra;
    Output result;
    ASSERT_EQ(model.postprocess({f32_output({1, 3, 256, 256}, 0.0f)}, ctx, result), StatusCode::OK);

    ASSERT_EQ(result.enhancement_result.size(), cv::Size(400, 300));
    ASSERT_EQ(result.enhancement_result.channels(), 4);
    std::vector<cv::Mat> planes;
    cv::split(result.enhancement_result, planes);
    ASSERT_EQ(planes.size(), 4u);
    for (int row = 0; row < 300; ++row) {
        for (int col = 0; col < 400; ++col) {
            EXPECT_EQ(planes[3].at<uchar>(row, col), bgra.at<cv::Vec4b>(row, col)[3]) << "alpha at " << row << "," << col;
        }
    }
}

TEST(EnlightenGanPreprocess, DynamicSizesUseRealCeil16) {
    // dynamic session (unset dims): the size follows the request aligned UP
    // to a multiple of 16. Integer division would floor (255 -> 240) and
    // crash on sub-16 inputs (10 -> 0), so the alignment must be a real ceil.
    TestModel model;
    model._m_input_dynamic = true;

    const cv::Mat mid(255, 255, CV_8UC3, cv::Scalar(90, 90, 90));
    auto tensors = model.preprocess(mid);
    ASSERT_EQ(tensors.size(), 2u);
    EXPECT_EQ(tensors[0].tensor.shape, (std::vector<int64_t>{1, 3, 256, 256}));

    const cv::Mat small(10, 10, CV_8UC3, cv::Scalar(90, 90, 90));
    tensors = model.preprocess(small);
    ASSERT_EQ(tensors.size(), 2u);
    EXPECT_EQ(tensors[0].tensor.shape, (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(tensors[1].tensor.shape, (std::vector<int64_t>{1, 1, 16, 16}));

    const cv::Mat tall(300, 400, CV_8UC3, cv::Scalar(90, 90, 90));
    tensors = model.preprocess(tall);
    ASSERT_EQ(tensors.size(), 2u);
    EXPECT_EQ(tensors[0].tensor.shape, (std::vector<int64_t>{1, 3, 304, 400}));
}

TEST(EnlightenGanPostprocess, GeometryComesFromTheRequestContext) {
    // postprocess must validate against context.network_size, never the model
    // member: the same instance serves requests of different sizes, and the
    // member no longer carries the current request's geometry
    TestModel model;
    model._m_input_size_host = cv::Size(256, 256); // stale on purpose
    model._m_input_dynamic = false;

    InferenceContext ctx;
    ctx.source_size = cv::Size(400, 300);   // a 300x400 (row x col) request
    ctx.network_size = cv::Size(400, 304);  // ran at 304x400 after 16-align
    Output result;
    ASSERT_EQ(model.postprocess({f32_output({1, 3, 304, 400}, 0.0f)}, ctx, result), StatusCode::OK);
    // the enhancement is decoded at network resolution, then scaled back to
    // the source size
    ASSERT_EQ(result.enhancement_result.size(), cv::Size(400, 300));

    // a context size that does not match the output tensor must fail: the
    // output is only ever interpreted at the geometry of its own request
    InferenceContext wrong_ctx = ctx;
    wrong_ctx.network_size = cv::Size(256, 256);
    EXPECT_NE(model.postprocess({f32_output({1, 3, 304, 400}, 0.0f)}, wrong_ctx, result), StatusCode::OK);
}
