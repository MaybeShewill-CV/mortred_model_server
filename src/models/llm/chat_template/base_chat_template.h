/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: base_chat_template.h
 * Date: 24-11-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_CHAT_TEMPLATE_H
#define MORTRED_MODEL_SERVER_BASE_CHAT_TEMPLATE_H

#include <string>
#include <utility>
#include <vector>

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace llm {
namespace chat_template {

struct ChatMessage {
    std::string role;
    std::string content;

    ChatMessage(std::string r, std::string c) : role(std::move(r)), content(std::move(c)) {}
};

class Dialog {
  public:
    Dialog() = default;

    ~Dialog() = default;

    Dialog(const Dialog &transformer) = default;

    Dialog& operator=(const Dialog &transformer) = default;

    explicit Dialog (const ChatMessage &msg) {
        messages.push_back(msg);
    }

    Dialog (const std::string& role, const std::string& content) {
        messages.emplace_back(role, content);
    }

    Dialog operator+(const Dialog& other) {
        Dialog tmp(*this);
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(tmp.messages));
        return tmp;
    }

    Dialog& operator+=(const Dialog& other) {
        std::copy(other.messages.begin(), other.messages.end(), std::back_inserter(messages));
        return *this;
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

  public:
    std::vector<ChatMessage> messages;
};

class BaseChatTemplate {
public:
    /***
    *
     */
    virtual ~BaseChatTemplate() = default;

    /***
     *
     * @param config
     */
    BaseChatTemplate() = default;

    /***
    *
    * @param transformer
     */
    BaseChatTemplate(const BaseChatTemplate& BaseChatTemplate) = default;

    /***
     *
     * @param transformer
     * @return
     */
    BaseChatTemplate& operator=(const BaseChatTemplate& transformer) = default;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    virtual jinq::common::StatusCode apply_chat_template(const Dialog& messages, std::string& out_fmt_str) = 0;
};
}
}
}
}

#endif // MORTRED_MODEL_SERVER_BASE_CHAT_TEMPLATE_H
