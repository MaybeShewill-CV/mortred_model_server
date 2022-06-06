/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: http_utils.cpp
* Date: 22-6-4
************************************************/

#include "http_utils.h"

#include <algorithm>
#include <random>
#include <future>
#include <fstream>

namespace morted {
namespace common {
namespace http_util {

/***
*
*/
MultipartParser::MultipartParser() {
    int i = 0;
    std::string rand_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int len = static_cast<int>(rand_chars.size());
    _m_boundary = "----CppRestSdkClient";

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

    while(i < 16) {
        int idx = dist(mt) % len;
        _m_boundary.push_back(rand_chars[idx]);
        ++i;
    }
}

/***
*
* @return
*/
const std::string &MultipartParser::gen_body_content() {
    std::vector<std::future<std::string> > futures;
    _m_body_content.clear();
    for(auto &file : _m_files) {
        std::future<std::string> content_futures = std::async(std::launch::async, [&file]() {
            std::ifstream ifile(file.second, std::ios::binary | std::ios::ate);
            std::streamsize size = ifile.tellg();
            ifile.seekg(0, std::ios::beg);
            char *buff = new char[size];
            ifile.read(buff, size);
            ifile.close();
            std::string ret(buff, size);
            delete[] buff;
            return ret;
        });
        futures.push_back(std::move(content_futures));
    }

    for(auto &param : _m_params) {
        _m_body_content += "\r\n--";
        _m_body_content += _m_boundary;
        _m_body_content += "\r\nContent-Disposition: form-data; name=\"";
        _m_body_content += param.first;
        _m_body_content += "\"\r\n\r\n";
        _m_body_content += param.second;
    }

    for(size_t i = 0; i < _m_files.size(); ++i) {
        std::string filename;
        std::string content_type;
        std::string file_content = futures[i].get();
        get_file_name_type(_m_files[i].second, filename, content_type);
        _m_body_content += "\r\n--";
        _m_body_content += _m_boundary;
        _m_body_content += "\r\nContent-Disposition: form-data; name=\"";
        _m_body_content += _m_files[i].first;
        _m_body_content += "\"; filename=\"";
        _m_body_content += filename;
        _m_body_content += "\"\r\nContent-Type: ";
        _m_body_content += content_type;
        _m_body_content += "\r\n\r\n";
        _m_body_content += file_content;
    }
    _m_body_content += "\r\n--";
    _m_body_content += _m_boundary;
    _m_body_content += "--\r\n";
    return _m_body_content;
}

/****** Private Function Sets ************/

/***
*
* @param file_path
* @param filename
* @param content_type
*/
void MultipartParser::get_file_name_type(
        const std::string &file_path,
        std::string& filename,
        std::string& content_type) {

    size_t last_spliter = file_path.find_last_of("/\\");
    filename = file_path.substr(last_spliter + 1);
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos == std::string::npos) {
        content_type = "application/octet-stream";
        return;
    }
    std::string ext = filename.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "jpg" || ext == "jpeg") {
        content_type = "image/jpeg";
    } else if (ext == "txt" || ext == "log") {
        content_type = "text/plain";
    } else {
        content_type = "application/octet-stream";
    }
    return;
}

}
}
}