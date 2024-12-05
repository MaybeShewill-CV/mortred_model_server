/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llama3.h
 * Date: 24-11-22
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_LLAMA3_H
#define MORTRED_MODEL_SERVER_LLAMA3_H

#include <memory>

#include "toml/toml.hpp"
#include "llama_cpp/llama.h"

#include "models/base_model.h"
#include "models/model_io_define.h"
#include "common/status_code.h"

namespace jinq {
namespace models {
namespace llm {
namespace llama {

struct ModelStatus {
    uint32_t n_ctx_size;
    int32_t kv_cache_cell_nums;
};

template <typename INPUT, typename OUTPUT> 
class Llama3 : public jinq::models::BaseAiModel<INPUT, OUTPUT> {
  public:
    /***
    * constructor
    * @param config
     */
    Llama3();

    /***
     *
     */
    ~Llama3() override;

    /***
    * constructor
    * @param transformer
     */
    Llama3(const Llama3 &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    Llama3 &operator=(const Llama3 &transformer) = delete;

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
    jinq::common::StatusCode run(const INPUT &input, OUTPUT &output) override;

    /***
     *
     * @param prompt
     * @param prompt_tokens
     * @return
     */
    jinq::common::StatusCode tokenize_prompt(const std::string &prompt, std::vector<llama_token> &prompt_tokens);

    /***
     *
     * @return
     */
    ModelStatus get_model_stat() const;

    /***
     *
     */
    void clear_kv_cache_cell() const;

    /***
     *
     * @param prompt
     * @return
     */
    int32_t count_prompt_token_nums(const std::string& prompt) const;

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

#include "llama3.inl"

#endif // MORTRED_MODEL_SERVER_LLAMA3_H
