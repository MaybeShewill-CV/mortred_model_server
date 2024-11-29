/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: Llama3ChatTemplate_chat_template.h
 * Date: 24-11-26
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LLAMA3_CHAT_TEMPLATE_CHAT_TEMPLATE_H
#define MORTRED_MODEL_SERVER_LLAMA3_CHAT_TEMPLATE_CHAT_TEMPLATE_H

#include <memory>

#include "models/llm/chat_template/base_chat_template.h"

namespace jinq {
namespace models {
namespace llm {
namespace chat_template {

class Llama3ChatTemplate : public BaseChatTemplate {
  public:
    /***
    * constructor
    * @param config
     */
    Llama3ChatTemplate();
    
    /***
     *
     */
    ~Llama3ChatTemplate() override;

    /***
    * constructor
    * @param transformer
     */
    Llama3ChatTemplate(const Llama3ChatTemplate &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    Llama3ChatTemplate &operator=(const Llama3ChatTemplate &transformer) = delete;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode apply_chat_template(const models::llm::chat_template::Dialog& dialog, std::string& out_fmt_str) override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}
}

#include "llama3_chat_template.inl"

#endif // MORTRED_MODEL_SERVER_LLAMA3_CHAT_TEMPLATE_CHAT_TEMPLATE_H
