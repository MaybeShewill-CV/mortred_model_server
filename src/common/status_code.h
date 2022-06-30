/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: status_code.h
* Date: 22-6-2
************************************************/

#ifndef MMAISERVER_STATUSCODE_H
#define MMAISERVER_STATUSCODE_H

#include <string>
#include <map>

namespace jinq {
namespace common {
enum StatusCode {
    OK = 0,

    // model status
    MODEL_INIT_FAILED = 1,
    MODEL_RUN_SESSION_FAILED = 2,
    MODEL_EMPTY_INPUT_IMAGE = 3,
    MODEL_RUN_TIMEOUT = 4,

    // server status
    SERVER_INIT_FAILED = 11,
    SERVER_RUN_FAILED = 12,

    // file status
    FILE_READ_ERROR = 30,
    FILE_WRITE_ERROR = 31,
    FILE_NOT_EXIST_ERROR = 32,

    // compress error
    COMPRESS_ERROR = 40,
    UNCOMPRESS_ERROR = 41,

    // json
    JSON_DECODE_ERROR = 50,
    JSON_ENCODE_ERROR = 51,
};

/***
 *
 * @param error_code
 * @return
 */
std::string error_code_to_str(int error_code);

}
}

#endif //MMAISERVER_STATUSCODE_H
