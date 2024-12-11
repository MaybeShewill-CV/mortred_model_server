/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: preprocess_wiki_corpus.cpp
 * Date: 24-12-10
 ************************************************/
// preprocess wiki corpus

#include <string>

#include <glog/logging.h>

#include "models/llm/rag//wiki_index_builder.h"

using jinq::models::llm::rag::WikiIndexBuilder;

int main(int argc, char** argv) {
    if (argc != 4) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path source_wiki_corpus_dir output_dir";
        return -1;
    }

    std::string cfg_path = argv[1];
    std::string source_wiki_corpus_dir = argv[2];
    std::string out_dir = argv[3];

    WikiIndexBuilder index_builder;
    auto cfg = toml::parse(cfg_path);
    index_builder.init(cfg);
    if (!index_builder.is_successfully_initialized()) {
        LOG(ERROR) << "init index builder failed";
        return -1;
    }

    auto status = index_builder.build_index(source_wiki_corpus_dir, out_dir);
    if (status != 0) {
        LOG(ERROR) << "build index failed";
    }

    return status;
}