/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: wiki_index_builder.h
 * Date: 24-12-9
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_WIKI_INDEX_BUILDER_H
#define MORTRED_MODEL_SERVER_WIKI_INDEX_BUILDER_H

#include <memory>

#include "toml/toml.hpp"
#include "faiss/IndexFlat.h"

#include "common/status_code.h"

namespace jinq {
namespace models {
namespace llm {
namespace rag {

class WikiIndexBuilder {
  public:
    /***
    * constructor
    * @param config
     */
    WikiIndexBuilder();
    
    /***
     *
     */
    ~WikiIndexBuilder();
    
    /***
    * constructor
    * @param transformer
     */
    WikiIndexBuilder(const WikiIndexBuilder &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    WikiIndexBuilder &operator=(const WikiIndexBuilder &transformer) = delete;

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
     * @param out_index_dir
     * @return
     */
    jinq::common::StatusCode build_index(const std::string& source_wiki_corpus_dir, const std::string& out_index_dir);

    /***
     *
     * @param index_f_path
     * @param index
     * @return
     */
    jinq::common::StatusCode load_index(const std::string& index_f_path);

    /***
     *
     * @param corpus_segment_path
     * @param segments
     * @return
     */
    jinq::common::StatusCode load_corpus_segment(const std::string& corpus_segment_path);

    /***
     *
     * @param input_prompt
     * @param referenced_corpus
     * @param top_k
     * @param aapply_chat_template
     * @return
     */
    jinq::common::StatusCode search(
        const std::string& input_prompt, std::string& referenced_corpus, int top_k=1, bool apply_chat_template=true);

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}
}

#endif // MORTRED_MODEL_SERVER_WIKI_INDEX_BUILDER_H
