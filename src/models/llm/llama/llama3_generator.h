/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: Llama3Generator_generator.h
 * Date: 24-11-28
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LLAMA3_GENERATOR_H
#define MORTRED_MODEL_SERVER_LLAMA3_GENERATOR_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "models/llm/base_generator.h"
#include "models/llm/llama/llama3.h"

namespace jinq {
namespace models {
namespace llm {
namespace llama {

class Llama3Generator : public BaseLlmGenerator {
  public:
    /***
    * constructor
    * @param config
     */
    Llama3Generator();
    
    /***
     *
     */
    ~Llama3Generator() override;

    /***
    * constructor
    * @param transformer
     */
    Llama3Generator(const Llama3Generator &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    Llama3Generator &operator=(const Llama3Generator &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse("")) &cfg) override;

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode text_completion(const std::string& prompt, OUT std::string& generate_output) override;

    /***
     *
     * @param dialog
     * @param generate_output
     * @param truncate
     * @return
     */
    jinq::common::StatusCode chat_completion(models::llm::chat_template::Dialog& dialog, OUT std::string& generate_output) override;

    /***
     *
     */
    void clear_kv_cache_cell();

    /***
     *
     * @return
     */
    llm::llama::ModelStatus get_model_stat() const;

    /***
     *
     * @param dialog
     * @return
     */
    int32_t count_dialog_token_nums(const models::llm::chat_template::Dialog& dialog) const;

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const override;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}
}

#endif // MORTRED_MODEL_SERVER_LLAMA3_GENERATOR_H
