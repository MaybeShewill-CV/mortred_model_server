/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: common_datatype.h
 * Date: 24-12-20
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_COMMON_DATATYPE_H
#define MORTRED_MODEL_SERVER_COMMON_DATATYPE_H

#include <string>
#include <vector>

#include "llama_cpp/llama.h"

namespace jinq {
namespace models {
namespace llm {

struct ModelStatus {
    uint32_t n_ctx_size;
    int32_t kv_cache_cell_nums;
    int32_t embed_dims;
};

class Dialog {
  public:
    /***
     *
     */
    Dialog() = default;

    /***
     *
     */
    ~Dialog() = default;

    /***
     *
     * @param transformer
     */
    Dialog(const Dialog &transformer) = default;

    /***
     *
     * @param transformer
     * @return
     */
    Dialog &operator=(const Dialog &transformer) = default;

    /***
     *
     * @param msg
     */
    explicit Dialog(const llama_chat_message &msg) { messages.push_back(msg); }

    /***
     *
     * @param role
     * @param content
     */
    Dialog(const std::string &role, const std::string &content) {
        llama_chat_message msg = {role.c_str(), content.c_str()};
        messages.push_back(msg);
    }

    /***
     *
     * @param role
     * @param content
     */
    Dialog(const char* role, const char* content) {
        llama_chat_message msg = {role, content};
        messages.push_back(msg);
    }

    /***
     *
     * @param other
     * @return
     */
    Dialog operator+(const Dialog &other) {
        Dialog tmp(*this);
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(tmp.messages));
        return tmp;
    }

    /***
     *
     * @param other
     * @return
     */
    Dialog &operator+=(const Dialog &other) {
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(messages));
        return *this;
    }

    /***
     *
     * @param index
     * @return
     */
    inline llama_chat_message &operator[](size_t index) { return messages[index]; }

    /***
     *
     * @param msg
     */
    inline void push_back(const llama_chat_message &msg) { messages.push_back(msg); }

    /***
     *
     */
    inline void clean_cache() { messages.clear(); }

    /***
     *
     * @return
     */
    bool empty() const { return messages.empty(); };

    /***
     *
     * @return
     */
    inline size_t size() const { return messages.size(); }

  public:
    std::vector<llama_chat_message> messages;
};

}
}
}

#endif // MORTRED_MODEL_SERVER_COMMON_DATATYPE_H
