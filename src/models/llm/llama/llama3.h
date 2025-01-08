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

#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "models/llm/llm_datatype.hpp"

namespace jinq {
namespace models {
namespace llm {

namespace llama {

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
    jinq::common::StatusCode init(const toml::value& cfg) override;

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
     * @param add_special
     * @return
     */
    jinq::common::StatusCode tokenize(const std::string &prompt, std::vector<llama_token> &prompt_tokens, bool add_special = true);

    /***
     *
     * @param prompt
     * @param out_embeddings
     * @param pool_type
     * @param truncated
     * @param max_seq_len
     * @param do_norm
     * @return
     */
    jinq::common::StatusCode get_embedding(
        const std::string& prompt, std::vector<std::vector<float> >& out_embeddings, const std::string& pool_type = "mean",
        bool truncated=true, int32_t max_seq_len=512, bool do_norm=true);

    /***
     *
     * @param input
     * @param output
     * @return
     */
    jinq::common::StatusCode text_completion(const std::string& prompt, std::string& generate_output);

    /***
     *
     * @param dialog
     * @param generate_output
     * @param truncate
     * @return
     */
    jinq::common::StatusCode chat_completion(Dialog& dialog, std::string& generate_output);

    /***
     *
     * @param dialog
     * @param add_ass
     * @param out_formatted_str
     * @return
     */
    jinq::common::StatusCode apply_chat_template(const Dialog& dialog, bool add_ass, std::string& out_formatted_str);

    /***
     *
     * @return
     */
    jinq::models::llm::ModelStatus get_model_stat() const;

    /***
     *
     */
    void clear_kv_cache_cell() const;

    /***
     *
     * @return
     */
    llama_perf_context_data get_context_perf() const;

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
