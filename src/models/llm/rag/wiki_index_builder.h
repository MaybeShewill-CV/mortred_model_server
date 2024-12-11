/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: WikiPreprocessor.h
 * Date: 24-12-9
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_WIKI_PREPROCESSOR_H
#define MORTRED_MODEL_SERVER_WIKI_PREPROCESSOR_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace llm {
namespace rag {
class WikiPreprocessor {
  public:
    /***
    * constructor
    * @param config
     */
    WikiPreprocessor();
    
    /***
     *
     */
    ~WikiPreprocessor();
    
    /***
    * constructor
    * @param transformer
     */
    WikiPreprocessor(const WikiPreprocessor &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    WikiPreprocessor &operator=(const WikiPreprocessor &transformer) = delete;

    /***
     *
     * @param cfg
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse("")) &cfg);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const;

    /***
     *
     * @param source_wiki_corpus_dir
     * @param out_save_path
     * @param chunk_word_size
     * @return
     */
    jinq::common::StatusCode chunk_wiki_corpus(
        const std::string& source_wiki_corpus_dir, const std::string& out_save_path, int chunk_word_size = 100);

    /***
     *
     * @param segmented_corpus_path
     * @param out_index_path
     * @return
     */
    jinq::common::StatusCode build_chunk_index(const std::string& segmented_corpus_path, const std::string& out_index_path);

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}
}

#endif // MORTRED_MODEL_SERVER_WIKI_PREPROCESSOR_H
