/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: qwen2_vl_chat_server.h
 * Date: 25-1-8
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_QWEN2_VL_CHAT_SERVER_H
#define MORTRED_MODEL_SERVER_QWEN2_VL_CHAT_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace llm {
namespace qwen2_vl {

class Qwen2VLChatServer : public jinq::server::BaseAiServer {
  public:
    /***
    * Constructor
    * @param config
     */
    Qwen2VLChatServer();

    /***
     *
     */
    ~Qwen2VLChatServer() override;

    /***
    * 赋值构造函数
    * @param transformer
     */
    Qwen2VLChatServer(const Qwen2VLChatServer &transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    Qwen2VLChatServer &operator=(const Qwen2VLChatServer &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse("")) &cfg) override;

    /***
     *
     * @param task
     */
    void serve_process(WFHttpTask *task) override;

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_impl;
};

}
}
}
}

#endif // MORTRED_MODEL_SERVER_QWEN2_VL_CHAT_SERVER_H
