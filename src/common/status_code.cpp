/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: status_code.cpp
* Date: 22-6-2
************************************************/

#include "status_code.h"

namespace jinq {
namespace common {

/***
 *
 * @param error_code
 * @return
 */
std::string error_code_to_str(int error_code) {
    switch (error_code) {
#define MORTRED_STATUS_CODE_TO_STR(name, value, desc) \
    case StatusCode::name: \
        return desc;
        MORTRED_STATUS_CODE_LIST(MORTRED_STATUS_CODE_TO_STR)
#undef MORTRED_STATUS_CODE_TO_STR
        default:
            return "Unknown";
    }
};
}
}
