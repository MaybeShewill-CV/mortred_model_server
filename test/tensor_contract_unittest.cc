#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "models/backend/tensor_contract.h"

using jinq::models::backend::DType;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorContract;
using jinq::models::backend::TensorLayout;

namespace {

NamedTensor named_f32(const std::string &name, const std::vector<int64_t> &shape, size_t byte_count, uint8_t fill = 0) {
    NamedTensor result;
    result.name = name;
    result.tensor.dtype = DType::F32;
    result.tensor.shape = shape;
    result.tensor.buffer.assign(byte_count, fill);
    return result;
}

} // namespace

TEST(TensorContract, AcceptsValidF32Tensor) {
    auto output = named_f32("output", {1, 2, 3}, 6 * sizeof(float));
    std::string error;
    EXPECT_TRUE(jinq::models::backend::validate_output_tensor(output, {DType::F32, 3, {1, -1, 3}}, &error)) << error;

    const float *data = nullptr;
    ASSERT_TRUE(jinq::models::backend::get_f32_data(output.tensor, &data, &error)) << error;
    ASSERT_TRUE(jinq::models::backend::require_finite_f32(data, 6, output.name, &error)) << error;
}

TEST(TensorContract, RejectsDtypeRankDimAndBuffer) {
    std::string error;
    auto output = named_f32("output", {1, 2, 3}, 6 * sizeof(float));
    output.tensor.buffer.assign(6 * sizeof(float), 0);

    auto bad_dtype = output;
    bad_dtype.tensor.dtype = DType::I32;
    EXPECT_FALSE(jinq::models::backend::validate_output_tensor(bad_dtype, {DType::F32, 3, {1, -1, 3}}, &error));
    EXPECT_NE(error.find("dtype"), std::string::npos);

    EXPECT_FALSE(jinq::models::backend::validate_output_tensor(output, {DType::F32, 2, {1, -1}}, &error));
    EXPECT_NE(error.find("rank"), std::string::npos);

    EXPECT_FALSE(jinq::models::backend::validate_output_tensor(output, {DType::F32, 3, {1, 2, 4}}, &error));
    EXPECT_NE(error.find("dim[2]"), std::string::npos);

    output.tensor.buffer.pop_back();
    EXPECT_FALSE(jinq::models::backend::validate_output_tensor(output, {DType::F32, 3, {1, -1, 3}}, &error));
    EXPECT_NE(error.find("buffer"), std::string::npos);
}

TEST(TensorContract, RejectsNonFiniteData) {
    auto output = named_f32("scores", {1, 3}, 3 * sizeof(float));
    output.tensor.buffer.assign(3 * sizeof(float), 0);
    auto *values = reinterpret_cast<float *>(output.tensor.buffer.data());
    values[1] = std::numeric_limits<float>::quiet_NaN();
    const float *data = nullptr;
    std::string error;
    ASSERT_TRUE(jinq::models::backend::get_f32_data(output.tensor, &data, &error));
    EXPECT_FALSE(jinq::models::backend::require_finite_f32(data, 3, output.name, &error));
    EXPECT_NE(error.find("non-finite"), std::string::npos);
}

TEST(TensorContract, FindsOutputByName) {
    std::vector<NamedTensor> outputs;
    outputs.push_back(named_f32("a", {1}, sizeof(float)));
    outputs.push_back(named_f32("b", {1}, sizeof(float)));
    ASSERT_NE(jinq::models::backend::find_output(outputs, "b"), nullptr);
    EXPECT_EQ(jinq::models::backend::find_output(outputs, "b")->name, "b");
    EXPECT_EQ(jinq::models::backend::find_output(outputs, "missing"), nullptr);
}

TEST(TensorContract, UnknownLayoutDoesNotCheckTag) {
    auto output = named_f32("output", {1, 3, 2, 2}, 12 * sizeof(float));
    output.tensor.layout = TensorLayout::Nhwc;
    std::string error;
    EXPECT_TRUE(jinq::models::backend::validate_output_tensor(
        output, {DType::F32, 4, {1, 3, 2, 2}}, &error))
        << error;
}

TEST(TensorContract, RejectsNhwcWhenContractIsNchw) {
    auto output = named_f32("output", {1, 3, 2, 2}, 12 * sizeof(float));
    output.tensor.layout = TensorLayout::Nhwc;
    std::string error;
    EXPECT_FALSE(jinq::models::backend::validate_output_tensor(
        output, {DType::F32, 4, {1, 3, 2, 2}, TensorLayout::Nchw}, &error));
    EXPECT_NE(error.find("layout"), std::string::npos) << error;
}

TEST(TensorContract, AcceptsMatchingNchwLayout) {
    auto output = named_f32("output", {1, 3, 2, 2}, 12 * sizeof(float));
    output.tensor.layout = TensorLayout::Nchw;
    std::string error;
    EXPECT_TRUE(jinq::models::backend::validate_output_tensor(
        output, {DType::F32, 4, {1, 3, 2, 2}, TensorLayout::Nchw}, &error))
        << error;
}

TEST(TensorLayout, HostOutputLayoutByRank) {
    EXPECT_EQ(jinq::models::backend::host_output_layout({1, 3, 8, 8}), TensorLayout::Nchw);
    EXPECT_EQ(jinq::models::backend::host_output_layout({1, 1, 120, 160}), TensorLayout::Nchw);
    EXPECT_EQ(jinq::models::backend::host_output_layout({1, 224, 224, 3}), TensorLayout::Nhwc);
    EXPECT_EQ(jinq::models::backend::host_output_layout({1, 1024, 2048, 3}), TensorLayout::Nhwc);
    EXPECT_EQ(jinq::models::backend::host_output_layout({1, 1000}), TensorLayout::Linear);
    EXPECT_EQ(jinq::models::backend::host_output_layout({1000}), TensorLayout::Linear);
    EXPECT_EQ(jinq::models::backend::host_output_layout({1, 84, 8400}), TensorLayout::Unknown);
    EXPECT_EQ(jinq::models::backend::host_output_layout({2, 3, 4, 5, 6}), TensorLayout::Unknown);
}

TEST(TensorContract, ShapeVolumeOverflowReturnsNegative) {
    using jinq::models::backend::shape_volume;
    EXPECT_EQ(shape_volume({2, 3, 4}), 24);
    EXPECT_EQ(shape_volume({0, 5}), 0);
    EXPECT_EQ(shape_volume({-1, 3}), -3);
    EXPECT_EQ(shape_volume({1LL << 40, 1LL << 40}), -1);
    EXPECT_EQ(shape_volume({std::numeric_limits<int64_t>::max(), 2}), -1);
}

TEST(TensorContract, CheckedShapeNbytesOverflow) {
    using jinq::models::backend::checked_shape_nbytes;
    size_t nbytes = 0;
    EXPECT_TRUE(checked_shape_nbytes({2, 3, 4}, DType::F32, &nbytes));
    EXPECT_EQ(nbytes, 2u * 3u * 4u * sizeof(float));
    EXPECT_FALSE(checked_shape_nbytes({0, 4}, DType::F32, &nbytes));
    EXPECT_FALSE(checked_shape_nbytes({-1, 4}, DType::F32, &nbytes));
    EXPECT_FALSE(checked_shape_nbytes({1LL << 40, 1LL << 40}, DType::F32, &nbytes));
}

TEST(TensorContract, CheckedElementCountOverflowPath) {
    Tensor tensor;
    tensor.dtype = DType::F32;
    tensor.shape = {std::numeric_limits<int64_t>::max()};
    size_t bytes = 0;
    std::string err;
    EXPECT_FALSE(jinq::models::backend::checked_element_count(tensor, &bytes, &err));
    EXPECT_NE(err.find("overflow"), std::string::npos) << err;
}

// A shape that is not concrete, has a zero dim, or overflows size_t used to be a
// CHECK abort (process death, every other in-flight request killed). It is now a
// std::invalid_argument that the model-layer fence (invoke_nothrow) converts into
// MODEL_RUN_SESSION_FAILED. Exact type match is required: no EXPECT_ANY_THROW.
TEST(TensorContract, MakeThrowsOnNonConcreteOrOverflowingShape) {
    EXPECT_THROW(Tensor::make(DType::F32, {-1, 3}), std::invalid_argument);
    EXPECT_THROW(Tensor::make(DType::F32, {0, 4}), std::invalid_argument);
    EXPECT_THROW(Tensor::make(DType::F32, {1LL << 40, 1LL << 40}), std::invalid_argument);
    EXPECT_THROW(Tensor::make(DType::I32, {std::numeric_limits<int64_t>::max(), 2}), std::invalid_argument);

    // the message carries the offending shape (shape_to_string)
    const std::vector<int64_t> bad_shape = {-1, 3};
    try {
        (void)Tensor::make(DType::F32, bad_shape);
        FAIL() << "Tensor::make accepted a non-concrete shape";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find(jinq::models::backend::shape_to_string(bad_shape)),
                  std::string::npos)
            << e.what();
    } catch (...) {
        FAIL() << "Tensor::make threw something other than std::invalid_argument";
    }

    // liveness proof: if any throw above had aborted the process this case would
    // never reach here; the happy path must still work after the failures.
    auto good = Tensor::make(DType::F32, {2, 3});
    EXPECT_EQ(good.byte_size(), 2u * 3u * sizeof(float));
    EXPECT_EQ(good.element_count(), 6);
    EXPECT_EQ(good.dtype, DType::F32);
}

TEST(TensorContract, DataThrowsOnDtypeMismatch) {
    auto f32 = Tensor::make(DType::F32, {1, 3});
    const Tensor &const_f32 = f32;

    EXPECT_THROW((void)f32.data<int32_t>(), std::invalid_argument);
    EXPECT_THROW((void)const_f32.data<int32_t>(), std::invalid_argument);
    EXPECT_THROW((void)const_f32.data<int64_t>(), std::invalid_argument);

    // the message names both the actual and the requested dtype
    try {
        (void)f32.data<int32_t>();
        FAIL() << "data<int32_t>() on an F32 tensor did not throw";
    } catch (const std::invalid_argument &e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("f32"), std::string::npos) << message;
        EXPECT_NE(message.find("i32"), std::string::npos) << message;
    } catch (...) {
        FAIL() << "data<int32_t>() threw something other than std::invalid_argument";
    }

    // liveness proof: the matched access still works in the same process
    float *mutable_values = f32.data<float>();
    ASSERT_NE(mutable_values, nullptr);
    mutable_values[0] = 1.5f;
    const float *values = const_f32.data<float>();
    ASSERT_NE(values, nullptr);
    EXPECT_FLOAT_EQ(values[0], 1.5f);

    auto i32 = Tensor::make(DType::I32, {1, 3});
    EXPECT_THROW((void)i32.data<float>(), std::invalid_argument);
}

// from_mat keeps its existing ok-flag convention; it must not be turned into a throw
TEST(TensorContract, FromMatContractUnchanged) {
    cv::Mat image(4, 5, CV_8UC3, cv::Scalar(7, 8, 9));
    bool ok = false;
    auto tensor = Tensor::from_mat(image, &ok);
    EXPECT_TRUE(ok);
    EXPECT_EQ(tensor.dtype, DType::U8);
    EXPECT_EQ(tensor.shape, (std::vector<int64_t>{1, 4, 5, 3}));
    EXPECT_EQ(tensor.byte_size(), 4u * 5u * 3u);
    EXPECT_EQ(tensor.layout, TensorLayout::Nhwc);
    ASSERT_NE(tensor.data<uint8_t>(), nullptr);
    EXPECT_EQ(tensor.data<uint8_t>()[0], 7);

    bool empty_ok = true;
    auto empty = Tensor::from_mat(cv::Mat(), &empty_ok);
    EXPECT_FALSE(empty_ok);
    EXPECT_EQ(empty.byte_size(), 0u);
}

