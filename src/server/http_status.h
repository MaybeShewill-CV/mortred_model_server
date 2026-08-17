/************************************************
 * Author: Codex
 * File: http_status.h
 * Date: 2026-08-26
 *
 * Maps business StatusCode to HTTP status codes.
 ************************************************/

#ifndef MORTRED_SERVER_HTTP_STATUS_H
#define MORTRED_SERVER_HTTP_STATUS_H

#include "common/status_code.h"

namespace jinq {
namespace server {

inline int http_status_of(jinq::common::StatusCode code) {
    using jinq::common::StatusCode;

    switch (code) {
        case StatusCode::OK:
            return 200;

        case StatusCode::JSON_DECODE_ERROR:
        case StatusCode::MODEL_EMPTY_INPUT_IMAGE:
            return 400;

        case StatusCode::UNSUPPORTED_MEDIA_TYPE:
            return 415;

        case StatusCode::REQUEST_ENTITY_TOO_LARGE:
            return 413;

        case StatusCode::UNAUTHORIZED:
            return 401;

        case StatusCode::RATE_LIMITED:
            return 429;

        case StatusCode::METHOD_NOT_ALLOWED:
            return 405;

        case StatusCode::NOT_FOUND:
            return 404;

        case StatusCode::MODEL_RUN_TIMEOUT:
            return 504;

        case StatusCode::MODEL_INIT_FAILED:
        case StatusCode::MODEL_RUN_SESSION_FAILED:
        case StatusCode::MODEL_EMPTY_OUTPUT:
        case StatusCode::SERVER_INIT_FAILED:
        case StatusCode::INTERNAL_ERROR:
        case StatusCode::TOKENIZE_UNKNOWN_TOKEN:
        case StatusCode::TRT_CUDA_ERROR:
        case StatusCode::TRT_ALLOC_MEMO_FAILED:
        case StatusCode::TRT_CONVERT_ONNX_MODEL_FAILED:
        case StatusCode::TRT_ALLOC_DYNAMIC_SHAPE_MEMO:
            return 500;

        default:
            return 500;
    }
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_SERVER_HTTP_STATUS_H
