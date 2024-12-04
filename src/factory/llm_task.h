/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llm_task.h
 * Date: 24-11-29
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LLM_TASK_H
#define MORTRED_MODEL_SERVER_LLM_TASK_H

#include "factory/base_factory.h"
#include "factory/register_marco.h"
#include "models/base_model.h"
#include "server/llm/llama/llama3_chat_server.h"

namespace jinq {
namespace factory {

using jinq::factory::ModelFactory;
using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

namespace llm {
namespace llama {

using jinq::server::llm::llama::Llama3ChatServer;

/***
 * create llama3 chat server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_llama3_chat_server(const std::string &server_name) {
    REGISTER_AI_SERVER(Llama3ChatServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}

}
}
}
}

#endif // MORTRED_MODEL_SERVER_LLM_TASK_H
