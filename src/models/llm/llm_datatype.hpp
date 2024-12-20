/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: common_datatype.h
 * Date: 24-12-20
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_COMMON_DATATYPE_H
#define MORTRED_MODEL_SERVER_COMMON_DATATYPE_H

#include <string>

namespace jinq {
namespace models {
namespace llm {

struct ChatMessage {
    std::string role;
    std::string content;
    ChatMessage(std::string r, std::string c) : role(std::move(r)), content(std::move(c)) {}
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
    Dialog& operator=(const Dialog &transformer) = default;

    /***
     *
     * @param msg
     */
    explicit Dialog (const ChatMessage &msg) {
        messages.push_back(msg);
    }

    /***
     *
     * @param role
     * @param content
     */
    Dialog (const std::string& role, const std::string& content) {
        messages.emplace_back(role, content);
    }

    /***
     *
     * @param other
     * @return
     */
    Dialog operator+(const Dialog& other) {
        Dialog tmp(*this);
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(tmp.messages));
        return tmp;
    }

    /***
     *
     * @param other
     * @return
     */
    Dialog& operator+=(const Dialog& other) {
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(messages));
        return *this;
    }

    inline ChatMessage& operator[](size_t index) {
        return messages[index];
    }

    inline void push_back(const ChatMessage& msg) {
        messages.push_back(msg);
    }

    inline void clean_cache() {
        messages.clear();
    }

    bool empty() const {
        return messages.empty();
    };

    inline size_t size() const {
        return messages.size();
    }

  public:
    std::vector<ChatMessage> messages;
};

struct ModelStatus {
    uint32_t n_ctx_size;
    int32_t kv_cache_cell_nums;
    int32_t embed_dims;
};


#endif // MORTRED_MODEL_SERVER_COMMON_DATATYPE_H
