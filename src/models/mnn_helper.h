/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: mnn_helper.h
* Date: 2026-08-14
************************************************/

#ifndef MORTRED_MODEL_SERVER_MNN_HELPER_H
#define MORTRED_MODEL_SERVER_MNN_HELPER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "MNN/Interpreter.hpp"

#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/model_config_schema.h"

namespace jinq {
namespace models {

/***
 * RAII MNN runtime. Owns the Interpreter/Session and caches input/output
 * tensors by name. init() parses the common MNN config block
 * (model_file_path, model_threads_num, compute_backend,
 * backend_precision_mode, backend_power_mode) and builds the session.
 * Teardown order: releaseSession -> releaseModel -> interpreter destructor.
 */
class MnnNet {
public:
    MnnNet() = default;
    MnnNet(const MnnNet&) = delete;
    MnnNet& operator=(const MnnNet&) = delete;

    ~MnnNet() {
        if (_m_net == nullptr) {
            return;
        }
        if (_m_session != nullptr) {
            _m_net->releaseSession(_m_session);
        }
        _m_net->releaseModel();
    }

    /***
     * build the session from the common config block and fetch the named tensors
     * @param cfg [SECTION] table of the model config
     * @param input_names input tensor names
     * @param output_names output tensor names
     * @return OK or MODEL_INIT_FAILED
     */
    jinq::common::StatusCode init(
        const toml::table& cfg,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names) {
        // contract check on the common MNN block: type errors fail fast,
        // model-specific keys are out of scope
        std::string schema_err;
        std::vector<std::string> schema_warnings;
        if (!validate_model_config_section(cfg, &schema_err, &schema_warnings)) {
            LOG(ERROR) << "invalid model config: " << schema_err;
            return jinq::common::StatusCode::MODEL_INIT_FAILED;
        }
        for (const auto& warning : schema_warnings) {
            LOG(WARNING) << warning;
        }

        if (!cfg.contains("model_file_path")) {
            LOG(ERROR) << "config does not have model_file_path field";
            return jinq::common::StatusCode::MODEL_INIT_FAILED;
        }
        const std::string model_file_path = cfg["model_file_path"].value_or<std::string>("");
        if (!jinq::common::FilePathUtil::is_file_exist(model_file_path)) {
            LOG(ERROR) << "model file not exist: " << model_file_path;
            return jinq::common::StatusCode::MODEL_INIT_FAILED;
        }

        _m_net.reset(MNN::Interpreter::createFromFile(model_file_path.c_str()));
        if (_m_net == nullptr) {
            LOG(ERROR) << "create MNN interpreter failed, model file: " << model_file_path;
            return jinq::common::StatusCode::MODEL_INIT_FAILED;
        }

        const int threads = static_cast<int>(cfg["model_threads_num"].value_or<int64_t>(4));
        if (threads <= 0) {
            LOG(WARNING) << "invalid model_threads_num: " << threads << ", use default 4";
            _m_threads = 4;
        } else {
            _m_threads = threads;
        }

        MNN::ScheduleConfig mnn_config;
        const std::string backend = cfg["compute_backend"].value_or<std::string>("cpu");
        if (backend == "cuda") {
            mnn_config.type = MNN_FORWARD_CUDA;
        } else if (backend == "cpu") {
            mnn_config.type = MNN_FORWARD_CPU;
        } else {
            LOG(WARNING) << "unsupported compute_backend: " << backend << ", use cpu";
            mnn_config.type = MNN_FORWARD_CPU;
        }
        mnn_config.numThread = _m_threads;

        MNN::BackendConfig backend_config;
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(
            cfg["backend_precision_mode"].value_or<int64_t>(MNN::BackendConfig::Precision_Normal));
        backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(
            cfg["backend_power_mode"].value_or<int64_t>(MNN::BackendConfig::Power_Normal));
        mnn_config.backendConfig = &backend_config;

        _m_session = _m_net->createSession(mnn_config);
        if (_m_session == nullptr) {
            LOG(ERROR) << "create MNN session failed, model file: " << model_file_path;
            _m_net->releaseModel();
            _m_net.reset();
            return jinq::common::StatusCode::MODEL_INIT_FAILED;
        }

        for (const auto& name : input_names) {
            MNN::Tensor* tensor = _m_net->getSessionInput(_m_session, name.c_str());
            if (tensor == nullptr) {
                LOG(ERROR) << "fetch input tensor failed: " << name;
                return jinq::common::StatusCode::MODEL_INIT_FAILED;
            }
            _m_inputs[name] = tensor;
        }
        for (const auto& name : output_names) {
            MNN::Tensor* tensor = _m_net->getSessionOutput(_m_session, name.c_str());
            if (tensor == nullptr) {
                LOG(ERROR) << "fetch output tensor failed: " << name;
                return jinq::common::StatusCode::MODEL_INIT_FAILED;
            }
            _m_outputs[name] = tensor;
        }

        return jinq::common::StatusCode::OK;
    }

    MNN::Tensor* input(const std::string& name) const {
        auto iter = _m_inputs.find(name);
        return iter == _m_inputs.end() ? nullptr : iter->second;
    }

    MNN::Tensor* output(const std::string& name) const {
        auto iter = _m_outputs.find(name);
        return iter == _m_outputs.end() ? nullptr : iter->second;
    }

    MNN::Interpreter* interpreter() const {
        return _m_net.get();
    }

    MNN::Session* session() const {
        return _m_session;
    }

    void run_session() const {
        _m_net->runSession(_m_session);
    }

    // dynamic input shape support: resize the tensor and re-allocate the session,
    // then re-fetch tensor pointers (MNN may re-allocate them on resize)
    void resize_tensor(MNN::Tensor* tensor, const std::vector<int>& dims) {
        _m_net->resizeTensor(tensor, dims);
        _m_net->resizeSession(_m_session);
        refresh_tensors();
    }

    void resize_tensor(MNN::Tensor* tensor, int batch, int channel, int height, int width) {
        _m_net->resizeTensor(tensor, batch, channel, height, width);
        _m_net->resizeSession(_m_session);
        refresh_tensors();
    }

    // first output tensor of the session (for models whose output node name
    // is unknown; e.g. fetched via getSessionOutputAll)
    MNN::Tensor* first_output() const {
        const auto& outputs = _m_net->getSessionOutputAll(_m_session);
        return outputs.empty() ? nullptr : outputs.begin()->second;
    }

    int threads() const {
        return _m_threads;
    }

private:
    void refresh_tensors() {
        for (auto& item : _m_inputs) {
            item.second = _m_net->getSessionInput(_m_session, item.first.c_str());
        }
        for (auto& item : _m_outputs) {
            item.second = _m_net->getSessionOutput(_m_session, item.first.c_str());
        }
    }

    std::unique_ptr<MNN::Interpreter> _m_net;
    MNN::Session* _m_session = nullptr;
    std::map<std::string, MNN::Tensor*> _m_inputs;
    std::map<std::string, MNN::Tensor*> _m_outputs;
    int _m_threads = 4;
};

}  // namespace models
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_MNN_HELPER_H
