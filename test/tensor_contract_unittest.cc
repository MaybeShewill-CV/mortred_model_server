#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "models/backend/tensor_contract.h"

using jinq::models::backend::DType;
using jinq::models::backend::NamedTensor;
using jinq::models::backend::Tensor;
using jinq::models::backend::TensorContract;

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
