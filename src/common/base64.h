/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base64.h
* Date: 22-6-2
************************************************/

#ifndef MM_AI_SERVER_BASE64_H
#define MM_AI_SERVER_BASE64_H

#include <string>

namespace morted {
namespace common {
class Base64 {
public:
    /***
     *
     */
    Base64() = delete;

    /***
     *
     */
    ~Base64() = default;

    /***
     * 赋值构造函数
     * @param transformer
     */
    Base64(const Base64 &transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    Base64 &operator=(const Base64 &transformer) = delete;

    /***
     * Base64 encode string
     * @param len
     * @return
     */
    static std::string base64_encode(unsigned char const* , unsigned int len);

    /***
     * Base64 encode string
     * @param input
     * @return
     */
    static std::string base64_encode(const std::string& input);

    /***
     * Base64 decode string
     * @param s
     * @return
     */
    static std::string base64_decode(const std::string& s);
};
}
}

#endif //MM_AI_SERVER_BASE64_H
