/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: status_code.cpp
* Date: 22-6-2
************************************************/

#include "status_code.h"

namespace morted {
namespace common {

namespace impl {
std::map<int, std::string> _m_error_code_table = {
    { StatusCode::OK, "OK" },
    { StatusCode::MODEL_INIT_FAILED, "Model init failed" },
    { StatusCode::COMPRESS_ERROR, "Compress Not Support" },
    { StatusCode::UNCOMPRESS_ERROR, "Uncompress Error" },
    { StatusCode::FILE_READ_ERROR, "File Read Error" },
    { StatusCode::FILE_WRITE_ERROR, "File Write Error" },
    { StatusCode::FILE_NOT_EXIST_ERROR, "File Not Exist Error" },
    { StatusCode::JSON_DECODE_ERROR, "Decode Json Error" },
    { StatusCode::JSON_ENCODE_ERROR, "Encode Json Error" },
};
}

/***
 *
 * @param error_code
 * @return
 */
std::string error_code_to_str(int error_code) {
    auto it = impl::_m_error_code_table.find(error_code);
    if(it == impl::_m_error_code_table.end()) {
        return "Unknown";
    } else {
        return it->second;
    }
};
}
}
