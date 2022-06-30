/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: FilePathUtil.h
* Date: 22-6-6
************************************************/

#ifndef MM_AI_SERVER_FILEPATHUTIL_H
#define MM_AI_SERVER_FILEPATHUTIL_H

#include <string>

namespace jinq {
namespace common {
class FilePathUtil {
public:
    /***
     *
     * @param path
     * @return
     */
    static bool is_dir_exist(const std::string& path);

    /***
     *
     * @param path
     * @return
     */
    static bool is_file_exist(const std::string& path);

    /***
     *
     * @param lhs
     * @param rhs
     * @return
     */
    static std::string concat_path(const std::string& lhs, const std::string& rhs);

    /***
     *
     * @param filepath
     * @return
     */
    static std::string get_file_name(const std::string& filepath);

    /***
     *
     * @param filepath
     * @return
     */
    static std::string get_file_suffix(const std::string& filepath);
};
}
}

#endif //MM_AI_SERVER_FILEPATHUTIL_H
