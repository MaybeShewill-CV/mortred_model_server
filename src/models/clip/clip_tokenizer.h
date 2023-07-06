/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: clip_tokenizer.h
 * Date: 23-7-6
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_CLIP_TOKENIZER_H
#define MORTRED_MODEL_SERVER_CLIP_TOKENIZER_H

#include <memory>
#include <vector>

#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace clip {

/***
 *
 */
class ClipTokenizer {
  public:
    /***
    * constructor
    * @param config
     */
    ClipTokenizer();

    /***
     *
     */
    ~ClipTokenizer();

    /***
    * constructor
    * @param transformer
     */
    ClipTokenizer(const ClipTokenizer& transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    ClipTokenizer& operator=(const ClipTokenizer& transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_text
     * @param text_embeddings
     * @return
     */
    jinq::common::StatusCode tokenize(const std::string& input_text, std::vector<int>& token);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
}
}
}

#endif // MORTRED_MODEL_SERVER_CLIP_TOKENIZER_H
