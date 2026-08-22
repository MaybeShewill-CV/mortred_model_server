/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: backend_cv_model.h
* Date: 26-8-20
************************************************/

#ifndef MORTRED_MODELS_BACKEND_BACKEND_CV_MODEL_H
#define MORTRED_MODELS_BACKEND_BACKEND_CV_MODEL_H

#include <memory>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "toml/toml.hpp"

#include "glog/logging.h"

#include "common/status_code.h"
#include "models/backend/backend_config.h"
#include "models/backend/session.h"
#include "models/backend/tensor.h"
#include "models/base_model.h"
#include "models/cv_image_input.h"

namespace jinq {
namespace models {
using jinq::common::StatusCode;

namespace detail {

template<typename INPUT, typename = void>
struct is_image_input : std::false_type {};

template<typename INPUT>
struct is_image_input<
    INPUT,
    std::void_t<decltype(cv_input::load_image(std::declval<const INPUT&>()))>>
    : std::true_type {};

inline bool has_extra_backend_table(const toml::table& model_section) {
    for (const auto& item : model_section) {
        const std::string key(item.first.str());
        if (key.size() > 8 && key.substr(key.size() - 8) == "_backend" &&
            item.second.is_table()) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

/***
 * Model author base class for the unified backend layer. It implements the
 * full init/run lifecycle:
 *
 *   init:      parse [SECTION.backend] -> create session -> on_init([SECTION.params])
 *   run_impl:  make_inputs -> preprocess -> session.run -> postprocess
 *
 * A standard single-image model only implements preprocess (cv::Mat to named
 * tensors) and postprocess (named tensors to task output). Non-image inputs
 * (clip tokens, image pairs, latent vectors) override make_inputs.
 *
 * The external BaseAiModel / factory contract is unchanged, so the server
 * layer is unaware of the backend selection.
 */
template<typename INPUT, typename OUTPUT>
class BackendCvModel : public BaseAiModel<INPUT, OUTPUT> {
  public:
    BackendCvModel(const BackendCvModel&) = delete;
    BackendCvModel& operator=(const BackendCvModel&) = delete;

    StatusCode init(const toml::table& cfg) final {
        _m_successfully_initialized = false;
        _m_session.reset();

        if (!cfg.contains(_m_section_name)) {
            LOG(ERROR) << "config section [" << _m_section_name << "] missing";
            return StatusCode::MODEL_INIT_FAILED;
        }
        const toml::table* model_section = cfg[_m_section_name].as_table();
        if (model_section == nullptr) {
            LOG(ERROR) << "config section [" << _m_section_name << "] missing or not a table";
            return StatusCode::MODEL_INIT_FAILED;
        }
        _m_model_section = *model_section;

        const bool has_primary_backend = model_section->contains("backend");
        const bool has_extra_backends = detail::has_extra_backend_table(*model_section);
        if (has_primary_backend) {
            std::string backend_err;
            if (!backend::parse_backend_config(*model_section, &_m_backend_config, &backend_err)) {
                LOG(ERROR) << "invalid backend config in [" << _m_section_name
                           << "]: " << backend_err;
                return StatusCode::MODEL_INIT_FAILED;
            }
            std::string session_err;
            _m_session = backend::InferenceSession::create(_m_backend_config, &session_err);
            if (_m_session == nullptr) {
                LOG(ERROR) << "create " << _m_backend_config.type << " session for ["
                           << _m_section_name << "] failed: " << session_err;
                return StatusCode::MODEL_INIT_FAILED;
            }
        } else if (!has_extra_backends) {
            LOG(ERROR) << "config section [" << _m_section_name
                       << "] has neither [backend] nor any [<name>_backend] sub-table";
            return StatusCode::MODEL_INIT_FAILED;
        }

        _m_params = toml::table{};
        if (model_section->contains("params")) {
            const toml::table* params = model_section->at("params").as_table();
            if (params == nullptr) {
                LOG(ERROR) << "[" << _m_section_name << ".params] must be a table";
                return StatusCode::MODEL_INIT_FAILED;
            }
            _m_params = *params;
        }

        const auto init_status = on_init(_m_params);
        if (init_status != StatusCode::OK) {
            LOG(ERROR) << "model specific init for [" << _m_section_name << "] failed";
            _m_session.reset();
            return init_status;
        }

        _m_successfully_initialized = true;
        return StatusCode::OK;
    }

    bool is_successfully_initialized() const final {
        // Multi-engine models own their sessions in derived state and have no
        // primary _m_session; init() clears this flag on every failure path.
        return _m_successfully_initialized;
    }

    /***
     * Generic smart batch - every single-session model gets real batching
     * for free. Packs the per-item make_inputs tensors positionally into
     * [N,...] tensors, runs ONE session, splits the outputs back by leading
     * dim and postprocesses per item (isolation contract as documented on
     * BaseAiModel::run_batch). When the engine rejects the batched shape
     * (e.g. a TRT engine built with a static batch-1 profile) or the output
     * layout is not batch-splittable, it transparently falls back to per-item
     * single runs: correct everywhere, faster wherever dynamic N is
     * supported. Multi-session models (lightglue / sam / clip) keep the
     * default per-item loop.
     */
    StatusCode run_batch(const std::vector<INPUT>& in,
                         std::vector<OUTPUT>& out,
                         std::vector<StatusCode>& item_status) override {
        if (_m_session == nullptr) {
            return BaseAiModel<INPUT, OUTPUT>::run_batch(in, out, item_status);
        }
        return run_image_batch(in, out, item_status);
    }

  protected:
    explicit BackendCvModel(std::string section_name)
        : _m_section_name(std::move(section_name)) {}

    ~BackendCvModel() override = default;

    /***
     * standard hook: single input image -> named input tensors. Models with
     * non-image inputs override make_inputs instead; the failing default
     * keeps such models from silently running an image through them.
     */
    virtual std::vector<backend::NamedTensor> preprocess(const cv::Mat& image) {
        (void)image;
        LOG(ERROR) << "model does not implement the single image preprocess path";
        return {};
    }

    /*** standard hook: named output tensors -> task output */
    virtual StatusCode postprocess(const std::vector<backend::NamedTensor>& outputs,
                                                 OUTPUT& output) = 0;

    /*** optional hook: model specific keys from [SECTION.params] */
    virtual StatusCode on_init(const toml::table& params) {
        (void)params;
        return StatusCode::OK;
    }

    /***
     * optional hook for non-image inputs (clip token ids, image pairs, ...).
     * The default loads the image through cv_input::load_image and forwards
     * it to preprocess. An empty return signals an invalid input.
     */
    virtual std::vector<backend::NamedTensor> make_inputs(const INPUT& input) {
        if constexpr (detail::is_image_input<INPUT>::value) {
            const cv::Mat image = cv_input::load_image(input);
            if (image.empty()) {
                LOG(ERROR) << "input image is empty";
                return {};
            }
            return preprocess(image);
        } else {
            LOG(ERROR) << "input type does not carry a loadable image, override make_inputs";
            return {};
        }
    }

    /*** the primary inference session created from [SECTION.backend] */
    backend::InferenceSession& session() {
        return *_m_session;
    }

    /***
     * Batch helpers for fixed-size NHWC image models: pack N preprocessed
     * CV_32FC3 HWC mats (identical H/W) into one [N,H,W,3] f32 input tensor.
     * Returns false on empty input or shape mismatch.
     */
    static bool pack_nhwc_batch(const std::string& input_name,
                                const std::vector<cv::Mat>& mats,
                                backend::NamedTensor* out) {
        if (out == nullptr || mats.empty()) {
            return false;
        }
        const int rows = mats.front().rows;
        const int cols = mats.front().cols;
        for (const auto& mat : mats) {
            if (mat.empty() || mat.type() != CV_32FC3 || mat.rows != rows || mat.cols != cols) {
                LOG(ERROR) << "batch preprocess mismatch: expected " << rows << "x" << cols
                           << " CV_32FC3 mats";
                return false;
            }
        }
        out->name = input_name;
        out->tensor = backend::Tensor::make<float>({static_cast<int64_t>(mats.size()), rows, cols, 3});
        const size_t item_bytes = static_cast<size_t>(rows) * cols * 3 * sizeof(float);
        for (size_t idx = 0; idx < mats.size(); ++idx) {
            std::memcpy(out->tensor.buffer.data() + idx * item_bytes, mats[idx].data, item_bytes);
        }
        return true;
    }

    /***
     * Slice a leading-dim batched output tensor into N per-item 1-D tensors
     * (contiguous copies). Classification postprocess reads scores by element
     * count, so the 1-D shape is sufficient; returns an empty vector when the
     * element count is not divisible by n.
     */
    static std::vector<backend::Tensor> split_batch_output(const backend::Tensor& tensor,
                                                           int64_t n) {
        std::vector<backend::Tensor> items;
        if (n <= 0 || tensor.element_count() <= 0 || tensor.element_count() % n != 0) {
            LOG(ERROR) << "cannot split batched output: elements=" << tensor.element_count()
                       << " n=" << n;
            return items;
        }
        const int64_t per = tensor.element_count() / n;
        const size_t per_bytes = static_cast<size_t>(per) * backend::dtype_size(tensor.dtype);
        items.reserve(static_cast<size_t>(n));
        for (int64_t idx = 0; idx < n; ++idx) {
            backend::Tensor item;
            item.dtype = tensor.dtype;
            item.shape = {per};
            item.buffer.resize(per_bytes);
            std::memcpy(item.buffer.data(), tensor.buffer.data() + idx * per_bytes, per_bytes);
            items.push_back(std::move(item));
        }
        return items;
    }

    /*** per-item fallback when a packed run is impossible (see run_batch) */
    static std::vector<size_t> indices_of(
        const std::vector<std::pair<size_t, std::vector<backend::NamedTensor>>>& prepared) {
        std::vector<size_t> indices;
        indices.reserve(prepared.size());
        for (const auto& entry : prepared) {
            indices.push_back(entry.first);
        }
        return indices;
    }

    StatusCode run_batch_fallback(const std::vector<INPUT>& inputs,
                                  const std::vector<size_t>& valid_items,
                                  std::vector<OUTPUT>& outputs,
                                  std::vector<StatusCode>& item_status) {
        StatusCode aggregate = StatusCode::OK;
        for (const size_t idx : valid_items) {
            item_status[idx] = run_impl(inputs[idx], outputs[idx]);
            if (item_status[idx] != StatusCode::OK) {
                aggregate = item_status[idx];
            }
        }
        return aggregate;
    }

    /*** generic packed-batch implementation shared by all single-session models */
    StatusCode run_image_batch(const std::vector<INPUT>& inputs,
                               std::vector<OUTPUT>& outputs,
                               std::vector<StatusCode>& item_status) {
        outputs.clear();
        item_status.assign(inputs.size(), StatusCode::OK);
        if (inputs.empty()) {
            LOG(ERROR) << "batch input is empty";
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        if (!is_successfully_initialized()) {
            LOG(ERROR) << "model is not successfully initialized, refuse to run batch";
            outputs.assign(inputs.size(), OUTPUT{});
            item_status.assign(inputs.size(), StatusCode::MODEL_INIT_FAILED);
            return StatusCode::MODEL_INIT_FAILED;
        }
        outputs.assign(inputs.size(), OUTPUT{});

        // 1) per-item inputs; a failing item is isolated and dropped
        std::vector<std::pair<size_t, std::vector<backend::NamedTensor>>> prepared;
        prepared.reserve(inputs.size());
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            auto tensors = make_inputs(inputs[idx]);
            if (tensors.empty()) {
                LOG(ERROR) << "batch item " << idx << ": input is empty";
                item_status[idx] = StatusCode::MODEL_EMPTY_INPUT_IMAGE;
                continue;
            }
            prepared.emplace_back(idx, std::move(tensors));
        }
        if (prepared.empty()) {
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }

        // 2) packing precondition: identical slot layout, leading dim == 1
        const auto& first_tensors = prepared.front().second;
        const size_t slot_count = first_tensors.size();
        for (const auto& [idx, tensors] : prepared) {
            (void)idx;
            if (tensors.size() != slot_count) {
                return run_batch_fallback(inputs, indices_of(prepared), outputs, item_status);
            }
            for (size_t slot = 0; slot < slot_count; ++slot) {
                const auto& a = first_tensors[slot].tensor;
                const auto& b = tensors[slot].tensor;
                if (a.shape != b.shape || a.shape.empty() || a.shape[0] != 1 ||
                    a.dtype != b.dtype) {
                    return run_batch_fallback(inputs, indices_of(prepared), outputs,
                                              item_status);
                }
            }
        }

        // 3) pack each slot into {N, rest...}
        const size_t batch_n = prepared.size();
        std::vector<backend::NamedTensor> batch_inputs(slot_count);
        for (size_t slot = 0; slot < slot_count; ++slot) {
            const auto& proto = first_tensors[slot].tensor;
            auto& packed = batch_inputs[slot];
            packed.name = first_tensors[slot].name;
            packed.tensor.dtype = proto.dtype;
            packed.tensor.shape = proto.shape;
            packed.tensor.shape[0] = static_cast<int64_t>(batch_n);
            const size_t item_bytes = proto.byte_size();
            packed.tensor.buffer.resize(item_bytes * batch_n);
            for (size_t pos = 0; pos < batch_n; ++pos) {
                const auto& src = prepared[pos].second[slot].tensor;
                std::memcpy(packed.tensor.buffer.data() + pos * item_bytes,
                            src.buffer.data(), item_bytes);
            }
        }

        // 4) one session run; engine rejection (static batch profile etc.)
        //    falls back to per-item runs
        std::vector<backend::NamedTensor> batch_outputs;
        if (_m_session->run(batch_inputs, batch_outputs) != StatusCode::OK) {
            LOG(INFO) << "packed batch rejected by the engine, falling back to "
                      << batch_n << " single runs";
            return run_batch_fallback(inputs, indices_of(prepared), outputs, item_status);
        }

        // 5) split: every output must carry the batch on its leading dim
        if (batch_outputs.size() == 0) {
            return run_batch_fallback(inputs, indices_of(prepared), outputs, item_status);
        }
        std::vector<std::vector<backend::Tensor>> per_slot_items(batch_outputs.size());
        for (size_t slot = 0; slot < batch_outputs.size(); ++slot) {
            const auto& tensor = batch_outputs[slot].tensor;
            if (tensor.shape.empty() || tensor.shape[0] != static_cast<int64_t>(batch_n)) {
                return run_batch_fallback(inputs, indices_of(prepared), outputs,
                                          item_status);
            }
            backend::Tensor item_proto;
            item_proto.dtype = tensor.dtype;
            item_proto.shape = tensor.shape;
            item_proto.shape[0] = 1;
            // byte_size() is buffer-based and the proto carries no buffer:
            // derive the item size from the concrete shape instead
            const size_t item_bytes =
                static_cast<size_t>(backend::shape_volume(item_proto.shape)) *
                backend::dtype_size(item_proto.dtype);
            if (item_bytes == 0 || tensor.buffer.size() < item_bytes * batch_n) {
                return run_batch_fallback(inputs, indices_of(prepared), outputs,
                                          item_status);
            }
            per_slot_items[slot].reserve(batch_n);
            for (size_t pos = 0; pos < batch_n; ++pos) {
                backend::Tensor item = item_proto;
                item.buffer.assign(tensor.buffer.begin() + static_cast<std::ptrdiff_t>(pos * item_bytes),
                                   tensor.buffer.begin() + static_cast<std::ptrdiff_t>((pos + 1) * item_bytes));
                per_slot_items[slot].push_back(std::move(item));
            }
        }

        // 6) per-item postprocess (isolation)
        StatusCode aggregate = StatusCode::OK;
        for (size_t pos = 0; pos < batch_n; ++pos) {
            const size_t idx = prepared[pos].first;
            std::vector<backend::NamedTensor> item_outputs;
            item_outputs.reserve(batch_outputs.size());
            for (size_t slot = 0; slot < batch_outputs.size(); ++slot) {
                item_outputs.push_back({batch_outputs[slot].name,
                                        per_slot_items[slot][pos]});
            }
            item_status[idx] = postprocess(item_outputs, outputs[idx]);
            if (item_status[idx] != StatusCode::OK) {
                aggregate = item_status[idx];
            }
        }
        return aggregate;
    }

    const backend::BackendConfig& backend_config() const {
        return _m_backend_config;
    }

    /*** model specific parameter table ([SECTION.params], may be empty) */
    const toml::table& params() const {
        return _m_params;
    }

    /*** the whole model section ([SECTION]), including *_backend sub-tables */
    const toml::table& model_section() const {
        return _m_model_section;
    }

    /***
     * build an additional session from a `<key>_backend` sub-table of the
     * model section, used by multi-engine models (lightglue, sam, ...)
     */
    std::unique_ptr<backend::InferenceSession> make_session(const std::string& backend_key) const {
        backend::BackendConfig extra_config;
        std::string err;
        if (!_m_model_section.contains(backend_key)) {
            LOG(ERROR) << "model section does not contain backend table [" << backend_key << "]";
            return nullptr;
        }
        const toml::table* backend_table = _m_model_section[backend_key].as_table();
        if (backend_table == nullptr) {
            LOG(ERROR) << "[" << backend_key << "] must be a table";
            return nullptr;
        }
        if (!backend::parse_backend_table(*backend_table, &extra_config, &err)) {
            LOG(ERROR) << "invalid backend table [" << backend_key << "]: " << err;
            return nullptr;
        }
        return backend::InferenceSession::create(extra_config, &err);
    }

    /***
     * multi-session orchestration hook. Models without a primary [backend]
     * table (lightglue, sam encoder+decoder, clip dual encoder, ...) override
     * this and drive their sessions themselves; the default implementation is
     * an error because it must never run silently.
     */
    virtual StatusCode run_sessions(const INPUT& input, OUTPUT& output) {
        (void)input;
        (void)output;
        LOG(ERROR) << "multi-session model does not implement run_sessions";
        return StatusCode::MODEL_RUN_SESSION_FAILED;
    }

    StatusCode run_impl(const INPUT& input, OUTPUT& output) final {
        if (_m_session == nullptr) {
            return run_sessions(input, output);
        }
        auto tensors = make_inputs(input);
        if (tensors.empty()) {
            return StatusCode::MODEL_EMPTY_INPUT_IMAGE;
        }
        std::vector<backend::NamedTensor> outputs;
        const auto run_status = _m_session->run(tensors, outputs);
        if (run_status != StatusCode::OK) {
            return run_status;
        }
        return postprocess(outputs, output);
    }

  private:
    std::string _m_section_name;
    backend::BackendConfig _m_backend_config;
    std::unique_ptr<backend::InferenceSession> _m_session;
    toml::table _m_model_section;
    toml::table _m_params;
    bool _m_successfully_initialized = false;
};

}  // namespace models
}  // namespace jinq

#endif  // MORTRED_MODELS_BACKEND_BACKEND_CV_MODEL_H
