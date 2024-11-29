/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: base_generator.h
 * Date: 24-11-28
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_GENERATOR_H
#define MORTRED_MODEL_SERVER_BASE_GENERATOR_H

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/llm/chat_template/base_chat_template.h"

namespace jinq {
namespace models {
namespace llm {

#define OUT
 
class BaseLlmGenerator {
  public:
    /***
    *
     */
    virtual ~BaseLlmGenerator() = default;

    /***
     * 
     * @param config
     */
    BaseLlmGenerator() = default;

    /***
    * 
    * @param transformer
     */
    BaseLlmGenerator(const BaseLlmGenerator &BaseLlmGenerator) = default;

    /***
     * 
     * @param transformer
     * @return
     */
    BaseLlmGenerator &operator=(const BaseLlmGenerator &transformer) = default;

    /***
     *
     * @param cfg
     * @return
     */
    virtual jinq::common::StatusCode init(const decltype(toml::parse("")) &cfg) = 0;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    virtual jinq::common::StatusCode text_completion(const std::string& prompt, OUT std::string& generate_output) = 0;

    /***
     *
     * @param dialog
     * @param generate_output
     * @return
     */
    virtual jinq::common::StatusCode chat_completion(models::llm::chat_template::Dialog& dialog, OUT std::string& generate_output) = 0;

    /***
     *
     * @return
     */
    virtual bool is_successfully_initialized() const = 0;
};
}
}
}

#endif // MORTRED_MODEL_SERVER_BASE_GENERATOR_H
