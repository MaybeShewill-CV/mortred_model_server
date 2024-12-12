/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: search_wiki_corpus.cpp
 * Date: 24-12-12
 ************************************************/
// search wiki corpus

#include <string>

#include <glog/logging.h>
#include "fmt/format.h"
#include "toml/toml.hpp"

#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/llm/rag/wiki_index_builder.h"

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::models::llm::rag::WikiIndexBuilder;

int main(int argc, char** argv) {
    if (argc != 6) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << "exe config_file_path index_path corpus_path query top_k";
        return -1;
    }

    std::string cfg_path = argv[1];
    std::string index_path = argv[2];
    std::string corpus_path = argv[3];
    std::string query = argv[4];
    int top_k = std::stoi(argv[5]);

    // init searcher
    auto cfg = toml::parse(cfg_path);
    WikiIndexBuilder searcher;
    searcher.init(cfg);
    if (!searcher.is_successfully_initialized()) {
        LOG(ERROR) << "init index builder&searcher failed";
        return -1;
    }

    // load corpus and index
    auto status = searcher.load_index(index_path);
    if (status != StatusCode::OJBK) {
        LOG(ERROR) << fmt::format("load index file: {} failed", index_path);
        return -1;
    }
    status = searcher.load_corpus_segment(corpus_path);
    if (status != StatusCode::OJBK) {
        LOG(ERROR) << fmt::format("load segment corpus file: {} failed", corpus_path);
        return -1;
    }

    // search top k corpus
    std::string reference_corpus;
    status = searcher.search(query, reference_corpus, top_k, true);
    if (status != StatusCode::OJBK) {
        LOG(ERROR) << fmt::format("search reference corpus failed status: {}", status);
        return -1;
    }
    LOG(INFO) << "successfully searched following reference corpus: ";
    LOG(INFO) << "---------------";
    LOG(INFO) << reference_corpus;

    return 1;
}