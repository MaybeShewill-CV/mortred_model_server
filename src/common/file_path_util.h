/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: file_path_util.h
* Date: 22-6-6
************************************************/

#ifndef MORTRED_MODEL_SERVER_FILE_PATH_UTIL_H
#define MORTRED_MODEL_SERVER_FILE_PATH_UTIL_H

#include <filesystem>
#include <string>
#include <system_error>

namespace jinq {
namespace common {

// thin wrappers over std::filesystem; edge cases (empty parts, separators,
// trailing slashes) are delegated to the standard library
class FilePathUtil {
public:
    static bool is_file_exist(const std::string& path) {
        std::error_code ec;
        return std::filesystem::is_regular_file(path, ec);
    }

    static std::string get_file_name(const std::string& filepath) {
        return std::filesystem::path(filepath).filename().string();
    }

    static std::string concat_path(const std::string& lhs, const std::string& rhs) {
        if (rhs.empty()) {
            return lhs;
        }
        return (std::filesystem::path(lhs) / std::filesystem::path(rhs))
            .lexically_normal()
            .string();
    }
};

}  // namespace common
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_FILE_PATH_UTIL_H
