#include "models/backend/model_runtime.h"

#include <algorithm>
#include <cstring>
#include <optional>

#include "common/cv_utils.h"
#include "glog/logging.h"

namespace jinq {
namespace models {
namespace backend {
namespace {

RuntimeStatus invalid_image(const std::string &operation, const std::string &message) {
    return runtime_error(StatusCode::MODEL_EMPTY_INPUT_IMAGE, "ImagePipeline " + operation + ": " + message);
}

bool set_error(StatusCode &status, std::string &destination, const RuntimeStatus &error) {
    if (status == StatusCode::OK) {
        status = error.status;
        destination = error.error;
    }
    return false;
}

bool valid_size(const cv::Size &size) { return size.width > 0 && size.height > 0; }

} // namespace

ImagePipeline::ImagePipeline(const cv::Mat &image) : image_(image.clone()) {
    if (image.empty()) {
        set_error(status_, error_, invalid_image("constructor", "input image is empty"));
    }
}

ImagePipeline &ImagePipeline::bgr_to_rgb() {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (image_.channels() != 3) {
        set_error(status_, error_, invalid_image("bgr_to_rgb", "expected a 3-channel image"));
        return *this;
    }
    cv::Mat converted;
    cv::cvtColor(image_, converted, cv::COLOR_BGR2RGB);
    image_ = std::move(converted);
    return *this;
}

ImagePipeline &ImagePipeline::bgra_to_rgb() {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (image_.channels() != 4) {
        set_error(status_, error_, invalid_image("bgra_to_rgb", "expected a 4-channel image"));
        return *this;
    }
    cv::Mat converted;
    cv::cvtColor(image_, converted, cv::COLOR_BGRA2RGB);
    image_ = std::move(converted);
    return *this;
}

ImagePipeline &ImagePipeline::bgr_to_gray() {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (image_.channels() != 3) {
        set_error(status_, error_, invalid_image("bgr_to_gray", "expected a 3-channel image"));
        return *this;
    }
    cv::Mat converted;
    cv::cvtColor(image_, converted, cv::COLOR_BGR2GRAY);
    image_ = std::move(converted);
    return *this;
}

ImagePipeline &ImagePipeline::rgb_to_bgr() {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (image_.channels() != 3) {
        set_error(status_, error_, invalid_image("rgb_to_bgr", "expected a 3-channel image"));
        return *this;
    }
    cv::Mat converted;
    cv::cvtColor(image_, converted, cv::COLOR_RGB2BGR);
    image_ = std::move(converted);
    return *this;
}

ImagePipeline &ImagePipeline::resize(const cv::Size &size) {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (!valid_size(size)) {
        set_error(status_, error_, invalid_image("resize", "target size must be positive"));
        return *this;
    }
    cv::Mat resized;
    cv::resize(image_, resized, size, 0.0, 0.0, cv::INTER_LINEAR);
    image_ = std::move(resized);
    return *this;
}

ImagePipeline &ImagePipeline::center_crop(const cv::Size &size) {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (!valid_size(size) || size.width > image_.cols || size.height > image_.rows) {
        set_error(status_, error_, invalid_image("center_crop", "crop does not fit the source image"));
        return *this;
    }
    const int x = (image_.cols - size.width) / 2;
    const int y = (image_.rows - size.height) / 2;
    image_ = image_(cv::Rect(x, y, size.width, size.height)).clone();
    return *this;
}

ImagePipeline &ImagePipeline::to_float() {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    cv::Mat converted;
    image_.convertTo(converted, CV_32F);
    image_ = std::move(converted);
    return *this;
}

ImagePipeline &ImagePipeline::scale(float factor) {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    image_ *= factor;
    return *this;
}

ImagePipeline &ImagePipeline::subtract(const std::array<float, 3> &values) {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (image_.channels() != 3) {
        set_error(status_, error_, invalid_image("subtract", "expected a 3-channel image"));
        return *this;
    }
    cv::subtract(image_, cv::Scalar(values[0], values[1], values[2]), image_);
    return *this;
}

ImagePipeline &ImagePipeline::divide(const std::array<float, 3> &values) {
    if (status_ != StatusCode::OK) {
        return *this;
    }
    if (image_.channels() != 3) {
        set_error(status_, error_, invalid_image("divide", "expected a 3-channel image"));
        return *this;
    }
    cv::divide(image_, cv::Scalar(values[0], values[1], values[2]), image_);
    return *this;
}

ImagePipeline &ImagePipeline::mean_std(const std::array<float, 3> &mean, const std::array<float, 3> &std) {
    subtract(mean);
    divide(std);
    return *this;
}

RuntimeResult<NamedTensor> ImagePipeline::pack(const std::string &name, bool nchw) const {
    if (status_ != StatusCode::OK) {
        return {status_, error_, {}};
    }
    if (name.empty()) {
        return {StatusCode::MODEL_EMPTY_INPUT_IMAGE, "ImagePipeline pack: tensor name is empty", {}};
    }
    if (image_.empty()) {
        return {StatusCode::MODEL_EMPTY_INPUT_IMAGE, "ImagePipeline pack: working image is empty", {}};
    }
    if (image_.type() != CV_32FC1 && image_.type() != CV_32FC3) {
        return {StatusCode::MODEL_EMPTY_INPUT_IMAGE, "ImagePipeline pack: expected CV_32FC1 or CV_32FC3", {}};
    }

    NamedTensor named;
    named.name = name;
    if (nchw) {
        named.tensor = Tensor::make<float>({1, image_.channels(), image_.rows, image_.cols});
    } else {
        named.tensor = Tensor::make<float>({1, image_.rows, image_.cols, image_.channels()});
    }

    if (nchw) {
        const auto data = jinq::common::CvUtils::convert_to_chw_vec(image_);
        if (data.size() * sizeof(float) != named.tensor.byte_size()) {
            return {StatusCode::MODEL_EMPTY_INPUT_IMAGE, "ImagePipeline pack: CHW data size mismatch", {}};
        }
        std::memcpy(named.tensor.buffer.data(), data.data(), named.tensor.byte_size());
    } else {
        if (image_.isContinuous()) {
            const auto bytes = image_.total() * image_.elemSize();
            if (bytes != named.tensor.byte_size()) {
                return {StatusCode::MODEL_EMPTY_INPUT_IMAGE, "ImagePipeline pack: HWC data size mismatch", {}};
            }
            std::memcpy(named.tensor.buffer.data(), image_.data, bytes);
        } else {
            auto *destination = named.tensor.buffer.data();
            const size_t row_bytes = static_cast<size_t>(image_.cols) * image_.elemSize();
            for (int row = 0; row < image_.rows; ++row) {
                std::memcpy(destination, image_.ptr(row), row_bytes);
                destination += row_bytes;
            }
        }
    }
    return runtime_ok(std::move(named));
}

RuntimeResult<NamedTensor> ImagePipeline::nchw(const std::string &name) const { return pack(name, true); }

RuntimeResult<NamedTensor> ImagePipeline::nhwc(const std::string &name) const { return pack(name, false); }

RuntimeResult<cv::Mat> ImagePipeline::mat() const {
    if (status_ != StatusCode::OK) {
        return {status_, error_, {}};
    }
    return runtime_ok(image_.clone());
}

OutputReader::OutputReader(const std::vector<NamedTensor> &outputs, std::string name) : outputs_(&outputs), name_(std::move(name)) {}

OutputReader &OutputReader::f32() {
    contract_.dtype = DType::F32;
    return *this;
}

OutputReader &OutputReader::shape(std::vector<int64_t> shape) {
    contract_.rank = shape.size();
    contract_.shape = std::move(shape);
    return *this;
}

OutputReader &OutputReader::finite() {
    require_finite_ = true;
    return *this;
}

RuntimeResult<F32OutputView> OutputReader::read() const {
    F32OutputView view;
    const auto status = validated_f32_named_output(*outputs_, name_, contract_, "OutputReader", &view);
    if (status != StatusCode::OK) {
        return {status, "OutputReader contract failed", {}};
    }
    return runtime_ok(std::move(view));
}

ParamReader::ParamReader(const toml::table &params, std::string log_prefix) : params_(params), prefix_(std::move(log_prefix)) {}

ParamReader &ParamReader::fail(const std::string &message) {
    if (status_.ok()) {
        status_ = runtime_error(StatusCode::MODEL_INIT_FAILED, prefix_ + ": " + message);
        LOG(ERROR) << status_.error;
    }
    return *this;
}

ParamReader &ParamReader::get(const std::string &key, int32_t *value) {
    last_key_ = key;
    if (status_.status != StatusCode::OK) {
        return *this;
    }
    if (!params_.contains(key) || !params_.at(key).is_integer()) {
        return fail("param '" + key + "' must be an integer");
    }
    const auto parsed = params_.at(key).value<int64_t>();
    if (!parsed.has_value()) {
        return fail("param '" + key + "' must be an integer");
    }
    *value = static_cast<int32_t>(*parsed);
    return *this;
}

ParamReader &ParamReader::get(const std::string &key, int64_t *value) {
    last_key_ = key;
    if (!status_.ok()) {
        return *this;
    }
    if (!params_.contains(key) || !params_.at(key).is_integer()) {
        return fail("param '" + key + "' must be an integer");
    }
    const auto parsed = params_.at(key).value<int64_t>();
    if (!parsed.has_value()) {
        return fail("param '" + key + "' must be an integer");
    }
    *value = *parsed;
    return *this;
}

ParamReader &ParamReader::get(const std::string &key, float *value) {
    double parsed = 0.0;
    const auto status = get(key, &parsed);
    if (status.ok()) {
        *value = static_cast<float>(parsed);
    }
    return *this;
}

ParamReader &ParamReader::get(const std::string &key, double *value) {
    last_key_ = key;
    if (!status_.ok()) {
        return *this;
    }
    if (!params_.contains(key) || (!params_.at(key).is_floating_point() && !params_.at(key).is_integer())) {
        return fail("param '" + key + "' must be numeric");
    }
    const auto parsed = params_.at(key).value<double>();
    if (!parsed.has_value()) {
        return fail("param '" + key + "' must be numeric");
    }
    *value = *parsed;
    return *this;
}

ParamReader &ParamReader::get(const std::string &key, bool *value) {
    last_key_ = key;
    if (!status_.ok()) {
        return *this;
    }
    if (!params_.contains(key) || !params_.at(key).is_boolean()) {
        return fail("param '" + key + "' must be boolean");
    }
    const auto parsed = params_.at(key).value<bool>();
    if (!parsed.has_value()) {
        return fail("param '" + key + "' must be boolean");
    }
    *value = *parsed;
    return *this;
}

ParamReader &ParamReader::get(const std::string &key, std::string *value) {
    last_key_ = key;
    if (!status_.ok()) {
        return *this;
    }
    if (!params_.contains(key) || !params_.at(key).is_string()) {
        return fail("param '" + key + "' must be a string");
    }
    const auto parsed = params_.at(key).value<std::string>();
    if (!parsed.has_value()) {
        return fail("param '" + key + "' must be a string");
    }
    *value = *parsed;
    return *this;
}

ParamReader &ParamReader::get(const std::string &key, cv::Size *value) {
    last_key_ = key;
    if (!status_.ok()) {
        return *this;
    }
    const toml::array *array = params_.contains(key) ? params_.at(key).as_array() : nullptr;
    if (array == nullptr || array->size() != 2) {
        return fail("param '" + key + "' must be [height, width]");
    }
    const auto height = (*array)[0].value<int64_t>();
    const auto width = (*array)[1].value<int64_t>();
    if (!height.has_value() || !width.has_value() || *height <= 0 || *width <= 0) {
        return fail("param '" + key + "' must contain positive integer dimensions");
    }
    *value = cv::Size(static_cast<int>(*width), static_cast<int>(*height));
    return *this;
}

ParamReader &ParamReader::min(double value) {
    if (!status_.ok()) {
        return *this;
    }
    const auto parsed = params_.contains(last_key_) ? params_.at(last_key_).value<double>() : std::optional<double>{};
    if (parsed.has_value() && *parsed < value) {
        fail("param '" + last_key_ + "' must be >= " + std::to_string(value));
    }
    return *this;
}

ParamReader &ParamReader::max(double value) {
    if (!status_.ok()) {
        return *this;
    }
    const auto parsed = params_.contains(last_key_) ? params_.at(last_key_).value<double>() : std::optional<double>{};
    if (parsed.has_value() && *parsed > value) {
        fail("param '" + last_key_ + "' must be <= " + std::to_string(value));
    }
    return *this;
}

ParamReader &ParamReader::non_empty() {
    if (!status_.ok()) {
        return *this;
    }
    const auto parsed = params_.contains(last_key_) ? params_.at(last_key_).value<std::string>() : std::optional<std::string>{};
    if (parsed.has_value() && parsed->empty()) {
        fail("param '" + last_key_ + "' must not be empty");
    }
    return *this;
}

ParamReader &ParamReader::array_size(size_t size) {
    if (!status_.ok()) {
        return *this;
    }
    const auto *array = params_.contains(last_key_) ? params_.at(last_key_).as_array() : nullptr;
    if (array != nullptr && array->size() != size) {
        fail("param '" + last_key_ + "' must have " + std::to_string(size) + " elements");
    }
    return *this;
}

ParamReader &ParamReader::allow_only_keys(const std::vector<std::string> &keys) {
    if (!status_.ok()) {
        return *this;
    }
    for (const auto &item : params_) {
        const std::string key(item.first.str());
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
            fail("unknown param '" + key + "'");
        }
    }
    return *this;
}

SessionIoValidator::SessionIoValidator(const InferenceSession &session) : session_(session) {}

SessionIoValidator &SessionIoValidator::input(const std::string &name) {
    use_output_ = false;
    name_ = name;
    return *this;
}

SessionIoValidator &SessionIoValidator::output(const std::string &name) {
    use_output_ = true;
    name_ = name;
    return *this;
}

SessionIoValidator &SessionIoValidator::dtype(DType value) {
    dtype_ = value;
    return *this;
}

SessionIoValidator &SessionIoValidator::f32() { return dtype(DType::F32); }

SessionIoValidator &SessionIoValidator::rank(size_t value) {
    rank_ = value;
    return *this;
}

SessionIoValidator &SessionIoValidator::shape(std::vector<int64_t> value) {
    expected_shape_ = std::move(value);
    rank_ = expected_shape_.size();
    return *this;
}

SessionIoValidator &SessionIoValidator::nchw() {
    has_layout_ = true;
    nchw_ = true;
    return *this;
}

SessionIoValidator &SessionIoValidator::nhwc() {
    has_layout_ = true;
    nchw_ = false;
    return *this;
}

SessionIoValidator &SessionIoValidator::channels(int64_t value) {
    channels_ = value;
    return *this;
}

SessionIoValidator &SessionIoValidator::static_shape() {
    static_shape_ = true;
    return *this;
}

SessionIoValidator &SessionIoValidator::allow_dynamic_batch() {
    allow_dynamic_batch_ = true;
    return *this;
}

RuntimeResult<TensorInfo> SessionIoValidator::validate() const {
    const auto &infos = use_output_ ? session_.outputs() : session_.inputs();
    const auto found = name_.empty()
                           ? infos.begin()
                           : std::find_if(infos.begin(), infos.end(), [this](const TensorInfo &item) { return item.name == name_; });
    if (found == infos.end()) {
        const std::string kind = use_output_ ? "output" : "input";
        const std::string name = name_.empty() ? "<first>" : name_;
        return {StatusCode::MODEL_INIT_FAILED, "session " + kind + " '" + name + "' is missing", {}};
    }
    const auto &info = *found;
    const std::string description = info.to_string();
    if (info.dtype != dtype_) {
        return {StatusCode::MODEL_INIT_FAILED, "unexpected session io dtype: " + description, {}};
    }
    if (rank_ != 0 && info.shape.size() != rank_) {
        return {StatusCode::MODEL_INIT_FAILED, "unexpected session io rank: " + description, {}};
    }
    if (expected_shape_.size() != 0) {
        if (expected_shape_.size() != info.shape.size()) {
            return {StatusCode::MODEL_INIT_FAILED, "unexpected session io rank: " + description, {}};
        }
        for (size_t idx = 0; idx < expected_shape_.size(); ++idx) {
            if (expected_shape_[idx] >= 0 && info.shape[idx] != expected_shape_[idx]) {
                return {StatusCode::MODEL_INIT_FAILED, "unexpected session io shape: " + description, {}};
            }
        }
    }
    if (channels_ >= 0) {
        if (info.shape.size() != 4 || (nchw_ ? info.shape[1] : info.shape[3]) != channels_) {
            return {StatusCode::MODEL_INIT_FAILED, "unexpected session io channels: " + description, {}};
        }
    }
    if (static_shape_ && info.dynamic) {
        return {StatusCode::MODEL_INIT_FAILED, "dynamic session io is not allowed: " + description, {}};
    }
    if (allow_dynamic_batch_) {
        for (size_t idx = 1; idx < info.shape.size(); ++idx) {
            if (info.shape[idx] <= 0) {
                return {StatusCode::MODEL_INIT_FAILED, "dynamic non-batch dimension: " + description, {}};
            }
        }
    }
    return runtime_ok(info);
}

} // namespace backend
} // namespace models
} // namespace jinq
