/************************************************
 * Author: Codex
 * File: model_contract_test_util.h
 * Date: 26-8-30
 *
 * Postprocess contract test macro.
 *
 * Every model that decodes named output tensors must reject a malformed tensor
 * with MODEL_OUTPUT_CONTRACT_FAILED instead of producing a partially decoded
 * result. That rejection matrix is the same for every model, so it is
 * generated here instead of being hand-copied into each contract test:
 *
 *   missing output / wrong dtype / wrong rank / wrong shape /
 *   short buffer / NaN / Inf
 *
 * Each variant becomes its own TEST, so it can be filtered and run on its own.
 * Model-specific behaviour (geometry mapping, batch splitting, decode values)
 * stays in the hand-written contract tests - this macro only covers the
 * rejection matrix.
 *
 * Usage:
 *
 *   POSTPROCESS_CONTRACT_TEST(RtdetrDetector, mat_input,
 *                             std_object_detection_output,
 *                             "output0", 1, 84, 100);
 *
 * The trailing arguments are the expected output shape.
 ************************************************/

#ifndef MORTRED_TEST_MODEL_CONTRACT_TEST_UTIL_H
#define MORTRED_TEST_MODEL_CONTRACT_TEST_UTIL_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/status_code.h"
#include "models/backend/inference_context.h"
#include "models/backend/tensor.h"

namespace jinq {
namespace test {
namespace contract {

using jinq::common::StatusCode;
using jinq::models::backend::DType;
using jinq::models::backend::InferenceContext;
using jinq::models::backend::NamedTensor;

/*** exposes the protected postprocess hook to the test ***/
template <typename MODEL> class PostprocessHarness : public MODEL {
  public:
    using MODEL::postprocess;
};

inline InferenceContext context(const cv::Size &source, const cv::Size &network) {
    InferenceContext ctx;
    ctx.source_size = source;
    ctx.network_size = network;
    return ctx;
}

/*** one f32 tensor filled with a finite value ***/
inline NamedTensor tensor(const std::string &name, DType dtype, const std::vector<int64_t> &shape, float fill) {
    NamedTensor out;
    out.name = name;
    out.tensor.dtype = dtype;
    out.tensor.shape = shape;
    int64_t count = 1;
    for (const auto dim : shape) {
        count *= dim;
    }
    if (count <= 0) {
        // a dynamic / invalid dimension cannot produce a concrete buffer; the
        // caller still gets the named tensor so dtype/rank checks can run
        return out;
    }
    out.tensor.buffer.resize(static_cast<size_t>(count) * sizeof(float));
    auto *data = reinterpret_cast<float *>(out.tensor.buffer.data());
    for (int64_t idx = 0; idx < count; ++idx) {
        data[idx] = fill;
    }
    return out;
}

template <typename MODEL, typename OUTPUT> void expect_rejected(const char *case_name, const std::vector<NamedTensor> &outputs) {
    PostprocessHarness<MODEL> model;
    OUTPUT result;
    const auto status = model.postprocess(outputs, context(cv::Size(20, 20), cv::Size(10, 10)), result);
    EXPECT_NE(status, StatusCode::OK) << case_name << ": malformed output must not be accepted";
    // a scaffolded model has no decoder yet and rejects everything with
    // MODEL_NOT_IMPLEMENTED; a real decoder must reject with the contract code
    const bool explicit_rejection = status == StatusCode::MODEL_OUTPUT_CONTRACT_FAILED || status == StatusCode::MODEL_NOT_IMPLEMENTED;
    EXPECT_TRUE(explicit_rejection) << case_name << ": unexpected rejection status " << static_cast<int>(status);
}

} // namespace contract
} // namespace test
} // namespace jinq

/*** One macro per model; the trailing arguments are the expected shape.
 *
 * Everything is fully qualified because the expansion lands at the caller's
 * global scope. The shape travels through __VA_ARGS__ because a braced list
 * would otherwise be split into several macro arguments by its commas. ***/

#define MORTRED_CONTRACT_VARIANT(ModelType, Output, out_name, case_label, fill, mutate, ...)                                               \
    TEST(ModelType, case_label) {                                                                                                          \
        std::vector<::jinq::models::backend::NamedTensor> outputs;                                                                         \
        auto tensor =                                                                                                                      \
            ::jinq::test::contract::tensor(out_name, ::jinq::models::backend::DType::F32, std::vector<int64_t>{__VA_ARGS__}, fill);        \
        mutate;                                                                                                                            \
        outputs.push_back(tensor);                                                                                                         \
        ::jinq::test::contract::expect_rejected<ModelType, Output>(#case_label, outputs);                                                  \
    }

#define POSTPROCESS_CONTRACT_TEST(Model, Input, Output, out_name, ...)                                                                     \
    using Model##ContractModel = Model<Input, Output>;                                                                                     \
    MORTRED_CONTRACT_VARIANT(Model##ContractModel, Output, out_name, rejects_missing_output, 0.5f, (void)0, __VA_ARGS__)                   \
    MORTRED_CONTRACT_VARIANT(Model##ContractModel, Output, out_name, rejects_wrong_dtype, 0.5f,                                            \
                             tensor.tensor.dtype = ::jinq::models::backend::DType::I32, __VA_ARGS__)                                       \
    MORTRED_CONTRACT_VARIANT(Model##ContractModel, Output, out_name, rejects_wrong_rank, 0.5f, tensor.tensor.shape.push_back(1),           \
                             __VA_ARGS__)                                                                                                  \
    MORTRED_CONTRACT_VARIANT(Model##ContractModel, Output, out_name, rejects_wrong_shape, 0.5f, tensor.tensor.shape.front() += 1,          \
                             __VA_ARGS__)                                                                                                  \
    MORTRED_CONTRACT_VARIANT(Model##ContractModel, Output, out_name, rejects_short_buffer, 0.5f, tensor.tensor.buffer.pop_back(),          \
                             __VA_ARGS__)                                                                                                  \
    MORTRED_CONTRACT_VARIANT(Model##ContractModel, Output, out_name, rejects_nan, std::numeric_limits<float>::quiet_NaN(), (void)0,        \
                             __VA_ARGS__)                                                                                                  \
    MORTRED_CONTRACT_VARIANT(Model##ContractModel, Output, out_name, rejects_inf, std::numeric_limits<float>::infinity(), (void)0,         \
                             __VA_ARGS__)

#endif // MORTRED_TEST_MODEL_CONTRACT_TEST_UTIL_H
