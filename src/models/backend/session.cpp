/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: session.cpp
 * Date: 26-8-20
 ************************************************/

#include "models/backend/session.h"

#include "glog/logging.h"

#include "models/backend/mnn_session.h"
#include "models/backend/ort_session.h"
#ifdef MORTRED_HAS_TRT
#include "models/backend/trt_session.h"
#endif

namespace jinq {
namespace models {
namespace backend {
using jinq::common::StatusCode;

std::unique_ptr<InferenceSession> InferenceSession::create(const BackendConfig& config,
                                                           std::string* err) {
    if (err != nullptr) {
        err->clear();
    }
    std::unique_ptr<InferenceSession> session;
    if (config.is_mnn()) {
        auto mnn_session = std::make_unique<MnnSession>();
        const auto status = mnn_session->init(config, err);
        if (status != StatusCode::OK) {
            return nullptr;
        }
        session = std::move(mnn_session);
    } else if (config.is_onnx()) {
        auto ort_session = std::make_unique<OrtSession>();
        const auto status = ort_session->init(config, err);
        if (status != StatusCode::OK) {
            return nullptr;
        }
        session = std::move(ort_session);
    } else if (config.is_tensorrt()) {
#ifdef MORTRED_HAS_TRT
        auto trt_session = std::make_unique<TrtSession>();
        const auto status = trt_session->init(config, err);
        if (status != StatusCode::OK) {
            return nullptr;
        }
        session = std::move(trt_session);
#else
        if (err != nullptr) {
            *err = "tensorrt backend is not compiled into this build (cpu profile); "
                   "use an mnn/onnx backend config or switch to the gpu profile";
        }
        return nullptr;
#endif
    } else {
        if (err != nullptr) {
            *err = "unknown backend type '" + config.type + "', expected mnn | onnx | tensorrt";
        }
        return nullptr;
    }
    return session;
}

}  // namespace backend
}  // namespace models
}  // namespace jinq
