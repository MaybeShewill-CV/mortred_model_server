/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: http_utils.h
* Date: 22-6-4
************************************************/

#ifndef MMAISERVER_HTTP_UTILS_H
#define MMAISERVER_HTTP_UTILS_H

#include <string>
#include <vector>

namespace jinq {
namespace common {
namespace http_util {
/***
 * multipart parser class
 */
class MultipartParser {
public:
    /***
     *
     */
    MultipartParser();

    /***
     *
     * @return
     */
    inline const std::string& body_content() {
        return _m_body_content;
    }

    /***
     *
     * @return
     */
    inline const std::string& boundary() {
        return _m_boundary;
    }

    /***
     *
     * @param name
     * @param value
     */
    inline void add_parameter(const std::string &name, const std::string &value) {
        _m_params.push_back(std::move(std::pair<std::string, std::string>(name, value)));
    }

    /***
     *
     * @param name
     * @param value
     */
    inline void add_file(const std::string &name, const std::string &value) {
        _m_files.push_back(std::move(std::pair<std::string, std::string>(name, value)));
    }

    /***
     *
     * @return
     */
    const std::string& gen_body_content();

private:
    void get_file_name_type(const std::string &file_path, std::string& filename, std::string& content_type);

private:
    std::string _m_boundary;
    std::string _m_body_content;
    std::vector<std::pair<std::string, std::string> > _m_params;
    std::vector<std::pair<std::string, std::string> > _m_files;
};

}
}
}

#endif //MMAISERVER_HTTP_UTILS_H
