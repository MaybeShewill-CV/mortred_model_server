/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: backend_cv_model.h
* Date: 26-8-20
************************************************/

#ifndef MORTRED_MODELS_BACKEND_BACKEND_CV_MODEL_H
#define MORTRED_MODELS_BACKEND_BACKEND_CV_MODEL_H

#include <memory>
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
