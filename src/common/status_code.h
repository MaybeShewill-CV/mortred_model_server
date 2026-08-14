/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: status_code.h
* Date: 22-6-2
************************************************/

#ifndef MORTRED_MODEL_SERVER_STATUSCODE_H
#define MORTRED_MODEL_SERVER_STATUSCODE_H

#include <string>

namespace jinq {
namespace common {

// single source of truth: enumerators, wire codes and messages are kept together.
// wire codes are part of the HTTP API contract and must stay stable.
#define MORTRED_STATUS_CODE_LIST(X) \
    X(OK, 0, "OK") \
    X(MODEL_INIT_FAILED, 1, "model init failed") \
    X(MODEL_RUN_SESSION_FAILED, 2, "model run session failed") \
    X(MODEL_EMPTY_INPUT_IMAGE, 3, "model input empty") \
    X(MODEL_RUN_TIMEOUT, 4, "model run timeout") \
    X(MODEL_EMPTY_OUTPUT, 5, "model output empty") \
    X(SERVER_INIT_FAILED, 11, "server init failed") \
    X(JSON_DECODE_ERROR, 50, "decode json error") \
    X(TOKENIZE_UNKNOWN_TOKEN, 80, "unknown token") \
    X(TRT_CUDA_ERROR, 90, "tensorrt cuda error") \
    X(TRT_ALLOC_MEMO_FAILED, 91, "tensorrt allocate memory failed") \
    X(TRT_CONVERT_ONNX_MODEL_FAILED, 92, "convert onnx model to trt failed") \
    X(TRT_ALLOC_DYNAMIC_SHAPE_MEMO, 93, "tensorrt allocate dynamic shape memory failed")

enum class StatusCode {
#define MORTRED_STATUS_CODE_DEFINE(name, value, desc) name = value,
    MORTRED_STATUS_CODE_LIST(MORTRED_STATUS_CODE_DEFINE)
#undef MORTRED_STATUS_CODE_DEFINE
};

// wire integer of a status code (stable HTTP API contract)
constexpr int to_underlying(StatusCode code) {
    return static_cast<int>(code);
}

// human readable message of a status code
inline std::string status_code_to_str(StatusCode code) {
    switch (code) {
#define MORTRED_STATUS_CODE_TO_STR(name, value, desc) \
    case StatusCode::name: \
        return desc;
        MORTRED_STATUS_CODE_LIST(MORTRED_STATUS_CODE_TO_STR)
#undef MORTRED_STATUS_CODE_TO_STR
    }
    return "Unknown";
}

}  // namespace common
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_STATUSCODE_H
