/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: preprocess_wiki_corpus.cpp
 * Date: 24-12-10
 ************************************************/
// preprocess wiki corpus

#include <string>

#include "models/llm/rag//wiki_index_builder.h"

using jinq::models::llm::rag::WikiPreprocessor;

int main(int argc, char** argv) {

    std::string cfg_path = argv[1];
    std::string source_wiki_corpus_dir = argv[2];

    WikiPreprocessor preprocessor;
    auto cfg = toml::parse(cfg_path);
    preprocessor.init(cfg);
//    preprocessor.chunk_wiki_corpus(source_wiki_corpus_dir, "tmp.json", 100);
    preprocessor.build_chunk_index("tmp.json", "index.engine");

    return 1;
}