/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: status_code.cpp
* Date: 22-6-2
************************************************/

#include "status_code.h"

#include <map>

namespace jinq {
namespace common {

namespace impl {

std::map<int, std::string>& get_error_code_table() {
    static std::map<int, std::string> m_error_code_table = {
        { StatusCode::OK, "OK" },

        { StatusCode::MODEL_INIT_FAILED, "model init failed" },
        { StatusCode::MODEL_RUN_TIMEOUT, "model run timeout" },
        { StatusCode::MODEL_EMPTY_INPUT_IMAGE, "model input empty" },
        { StatusCode::MODEL_RUN_SESSION_FAILED, "model run session failed" },

        { StatusCode::SERVER_INIT_FAILED, "server init failed" },
        { StatusCode::SERVER_RUN_FAILED, "server run failed" },

        { StatusCode::FILE_READ_ERROR, "file read error" },
        { StatusCode::FILE_WRITE_ERROR, "file write error" },
        { StatusCode::FILE_NOT_EXIST_ERROR, "file not exist error" },

        { StatusCode::COMPRESS_ERROR, "compress not support" },
        { StatusCode::UNCOMPRESS_ERROR, "uncompress error" },

        { StatusCode::JSON_DECODE_ERROR, "decode json error" },
        { StatusCode::JSON_ENCODE_ERROR, "encode sson error" },

        { StatusCode::MYSQL_INIT_DB_CONFIG_ERROR, "init mysql connection failed"},
        { StatusCode::MYSQL_SELECT_FAILED, "exec select sql failed"},
        { StatusCode::MYSQL_INSERT_FAILED, "exec insert sql failed"},
        { StatusCode::MYSQL_UPDATE_FAILED, "exec update sql failed"},
        { StatusCode::MYSQL_DELETE_FAILED, "exec delete sql failed"},

    };
    return m_error_code_table;
}
}

/***
 *
 * @param error_code
 * @return
 */
std::string error_code_to_str(int error_code) {
    auto it = impl::get_error_code_table().find(error_code);

    if (it == impl::get_error_code_table().end()) {
        return "Unknown";
    } else {
        return it->second;
    }
};
}
}
