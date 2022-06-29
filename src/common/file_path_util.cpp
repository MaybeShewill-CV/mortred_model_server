/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: FileFilePathUtil.cpp
* Date: 22-6-6
************************************************/

#include "file_path_util.h"

#ifdef _WIN32
#include <io.h>
#endif
#include <sys/stat.h>

namespace mortred {
namespace common {
    
std::string FilePathUtil::get_file_name(const std::string& filepath) {
    std::string::size_type pos1 = filepath.find_last_not_of('/');
    if (pos1 == std::string::npos) {
        return "";
    }

    std::string::size_type pos2 = filepath.find_last_of('/', pos1);
    if (pos2 == std::string::npos) {
        pos2 = 0;
    } else {
        pos2++;
    }

    return filepath.substr(pos2, pos1 - pos2 + 1);
}

std::string FilePathUtil::get_file_suffix(const std::string& filepath) {
    std::string::size_type pos1 = filepath.find_last_of('/');
    if (pos1 == std::string::npos) {
        pos1 = 0;
    } else {
        pos1++;
    }

    std::string file = filepath.substr(pos1, -1);

    std::string::size_type pos2 = file.find_last_of('.');

    if (pos2 == std::string::npos) {
        return "";
    }

    return file.substr(pos2 + 1, -1);
}

std::string FilePathUtil::concat_path(const std::string& lhs, const std::string& rhs) {
    std::string res;
    if (lhs.back() == '/' && rhs.front() == '/') {
        res = lhs.substr(0, lhs.size() - 1) + rhs;
    } else if (lhs.back() != '/' && rhs.front() != '/') {
        res = lhs + "/" + rhs;
    } else {
        res = lhs + rhs;
    }

    return res;
}

bool FilePathUtil::is_dir_exist(const std::string& path) {
#ifdef _WIN32
    int n_ret = _access(path.c_str(), 0);
    return n_ret == 0;
#else
    struct stat st{};
    return stat(path.c_str(), &st) >= 0 && S_ISDIR(st.st_mode);
#endif
}

bool FilePathUtil::is_file_exist(const std::string& path) {
#ifdef _WIN32
    int n_ret = _access(path.c_str(), 0);
    return n_ret == 0;
#else
    struct stat st{};
    return stat(path.c_str(), &st) >= 0 && S_ISREG(st.st_mode);
#endif
}
}
}
