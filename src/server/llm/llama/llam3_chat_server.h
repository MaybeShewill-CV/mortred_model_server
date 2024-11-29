/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llam3_chat_server.h
 * Date: 24-11-29
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LLAM3_CHAT_SERVER_H
#define MORTRED_MODEL_SERVER_LLAM3_CHAT_SERVER_H

#include <memory>

#include "server/abstract_server.h"

namespace jinq {
namespace server {
namespace llm {
namespace llama {

class Llama3ChatServer : public jinq::server::BaseAiServer {
  public:
    /***
    * Constructor
    * @param config
     */
    Llama3ChatServer();

    /***
     *
     */
    ~Llama3ChatServer() override;

    /***
    * 赋值构造函数
    * @param transformer
     */
    Llama3ChatServer(const Llama3ChatServer &transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    Llama3ChatServer &operator=(const Llama3ChatServer &transformer) = delete;

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

#endif // MORTRED_MODEL_SERVER_LLAM3_CHAT_SERVER_H
