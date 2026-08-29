#ifndef MORTRED_MODELS_BACKEND_MODEL_RUNTIME_H
#define MORTRED_MODELS_BACKEND_MODEL_RUNTIME_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/status_code.h"
#include "models/backend/f32_output.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "toml/toml.hpp"

namespace jinq {
namespace models {
namespace backend {

using jinq::common::StatusCode;

/*** status-only result used by operations that do not produce a value */
struct RuntimeStatus {
    StatusCode status = StatusCode::OK;
    std::string error;

    bool ok() const { return status == StatusCode::OK; }
};

template <typename T> struct RuntimeResult {
    StatusCode status = StatusCode::OK;
    std::string error;
    T value{};

    bool ok() const { return status == StatusCode::OK; }
};

template <typename T> RuntimeResult<T> runtime_ok(T value) { return {StatusCode::OK, {}, std::move(value)}; }

inline RuntimeStatus runtime_error(StatusCode status, std::string error) { return {status, std::move(error)}; }

/***
 * Fluent preprocessing pipeline for ordinary image models. The pipeline owns a
 * private working Mat and never mutates the caller's source image.
 */
class ImagePipeline {
  public:
    explicit ImagePipeline(const cv::Mat &image);

    ImagePipeline &bgr_to_rgb();
    ImagePipeline &rgb_to_bgr();
    ImagePipeline &resize(const cv::Size &size);
    ImagePipeline &center_crop(const cv::Size &size);
    ImagePipeline &to_float();
    ImagePipeline &scale(float factor);
    ImagePipeline &subtract(const std::array<float, 3> &values);
    ImagePipeline &divide(const std::array<float, 3> &values);
    ImagePipeline &mean_std(const std::array<float, 3> &mean, const std::array<float, 3> &std);

    RuntimeResult<NamedTensor> nchw(const std::string &name) const;
    RuntimeResult<NamedTensor> nhwc(const std::string &name) const;
    RuntimeResult<cv::Mat> mat() const;

  private:
    RuntimeResult<NamedTensor> pack(const std::string &name, bool nchw) const;

    cv::Mat image_;
    StatusCode status_ = StatusCode::OK;
    std::string error_;
};

/*** fluent reader around the existing named-f32 output contract */
class OutputReader {
  public:
    OutputReader(const std::vector<NamedTensor> &outputs, std::string name);

    OutputReader &f32();
    OutputReader &shape(std::vector<int64_t> shape);
    OutputReader &finite();

    RuntimeResult<F32OutputView> read() const;

  private:
    const std::vector<NamedTensor> *outputs_ = nullptr;
    std::string name_;
    TensorContract contract_;
    bool require_finite_ = false;
};

/*** small TOML reader with unified type/range diagnostics */
class ParamReader {
  public:
    ParamReader(const toml::table &params, std::string log_prefix);

    ParamReader &get(const std::string &key, int32_t *value);
    ParamReader &get(const std::string &key, int64_t *value);
    ParamReader &get(const std::string &key, float *value);
    ParamReader &get(const std::string &key, double *value);
    ParamReader &get(const std::string &key, bool *value);
    ParamReader &get(const std::string &key, std::string *value);
    ParamReader &get(const std::string &key, cv::Size *value);

    ParamReader &min(double value);
    ParamReader &max(double value);
    ParamReader &non_empty();
    ParamReader &array_size(size_t size);
    ParamReader &allow_only_keys(const std::vector<std::string> &keys);

    RuntimeStatus status() const { return status_; }
    bool ok() const { return status_.ok(); }

  private:
    ParamReader &fail(const std::string &message);

    const toml::table &params_;
    std::string prefix_;
    RuntimeStatus status_;
    std::string last_key_;
};

/*** validates an InferenceSession IO tensor before a model caches its shape */
class SessionIoValidator {
  public:
    explicit SessionIoValidator(const InferenceSession &session);

    SessionIoValidator &input(const std::string &name = {});
    SessionIoValidator &output(const std::string &name = {});
    SessionIoValidator &f32();
    SessionIoValidator &dtype(DType value);
    SessionIoValidator &rank(size_t value);
    SessionIoValidator &shape(std::vector<int64_t> value);
    SessionIoValidator &nchw();
    SessionIoValidator &nhwc();
    SessionIoValidator &channels(int64_t value);
    SessionIoValidator &static_shape();
    SessionIoValidator &allow_dynamic_batch();

    RuntimeResult<TensorInfo> validate() const;

  private:
    const InferenceSession &session_;
    bool use_output_ = false;
    std::string name_;
    DType dtype_ = DType::F32;
    size_t rank_ = 0;
    std::vector<int64_t> expected_shape_;
    bool has_layout_ = false;
    bool nchw_ = false;
    int64_t channels_ = -1;
    bool static_shape_ = false;
    bool allow_dynamic_batch_ = false;
};

} // namespace backend
} // namespace models
} // namespace jinq

#endif // MORTRED_MODELS_BACKEND_MODEL_RUNTIME_H
