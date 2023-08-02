/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: clip_tokenizer.cpp
 * Date: 23-7-6
 ************************************************/

#include "clip_tokenizer.h"

#include <iostream>
#include <regex>
#include <fstream>
#include <unordered_map>

#include "glog/logging.h"

#include "common/file_path_util.h"
#include "common/time_stamp.h"

namespace jinq {
namespace models {

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using jinq::common::Timestamp;

namespace clip {

using id = int32_t;
using token = std::string;

class ClipTokenizer::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param input_text
     * @param token
     * @return
     */
    StatusCode tokenize(const std::string& input_text, std::vector<int>& token);

    /***
     * if model successfully initialized
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_init_model;
    }

  private:
    // model file path
    std::string _m_model_path;

    int _m_vocab_counts = 49408;

    std::unordered_map<token, id> _m_token_to_id;
    std::unordered_map<id, token> _m_id_to_token;
    std::vector<std::string> _m_special_tokens;

    // init flag
    bool _m_successfully_init_model = false;

    std::string convert_to_utf8(const std::wstring &input);

    std::wstring convert_to_wstring(const std::string &input);

    void add_special_token(const std::string &token);
};

/************ Impl Implementation ************/

/***
 *
 * @param cfg
 * @return
 */
StatusCode ClipTokenizer::Impl::init(const decltype(toml::parse("")) &cfg) {

    auto token_cfg = cfg.at("TOKENIZER");

    auto vocab_file_path = token_cfg["vocab_file_path"].as_string();
    if (!FilePathUtil::is_file_exist(vocab_file_path)) {
        LOG(ERROR) << "vocab file path: " << vocab_file_path << " not exists";
        _m_successfully_init_model = false;
        return StatusCode::MODEL_INIT_FAILED;
    }

    auto fin = std::ifstream(vocab_file_path, std::ios::in);

    std::string word;
    int i = 0;
    while (std::getline(fin, word)) {
        _m_token_to_id[word] = i;
        _m_id_to_token[i] = word;
        i++;
    }

    _m_successfully_init_model = true;
    LOG(INFO) << "Successfully clip tokenizer";
    return StatusCode::OJBK;
}

/***
 *
 * @param input_image
 * @param token
 * @return
 */
StatusCode ClipTokenizer::Impl::tokenize(const std::string& input_text, std::vector<int32_t> &tokens) {
    std::vector<std::string> words;
    // first split the text into words
    std::string str = input_text;
    std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

    // generate the subpattern from the special_tokens vector if it's not empty
    if (!_m_special_tokens.empty()) {
        std::string special_tokens_subpattern;
        for (const auto &token : _m_special_tokens) {
            if (!special_tokens_subpattern.empty()) {
                special_tokens_subpattern += "|";
            }
            special_tokens_subpattern += token;
        }

        // Modify the regex pattern with the generated special tokens subpattern
        pat = special_tokens_subpattern + "|" + pat;
    }

    std::regex re(pat);
    std::smatch m;
    while (std::regex_search(str, m, re)) {
        for (auto x : m) {
            words.push_back(x);
        }
        str = m.suffix();
    }

    tokens.push_back(49406); // start of text
    for (const auto &word : words) {
        std::string full_word;
        if (word.rfind(" ", 0) == 0) {
            full_word += word.substr(1);
        } else {
            full_word += word;
        }
        full_word += "</w>";
        auto wit = _m_token_to_id.find(full_word);
        if (wit != _m_token_to_id.end()) {
            tokens.push_back(wit->second);
            continue;
        }

        for (size_t i = 0; i < word.size();) {
            for (size_t j = word.size() - 1; j >= i; j--) {
                auto cand = word.substr(i, j - i + 1);
                auto it = _m_token_to_id.find(cand);
                if (it != _m_token_to_id.end()) { // word.substr(i, j-i+1) in vocab
                    tokens.push_back(it->second);
                    i = j + 1;
                    break;
                } else if (j == i) {
                    LOG(ERROR) << "unknown token: " << word.substr(i, 1).data();
                    i++;
                    return StatusCode::TOKENIZE_UNKNOWN_TOKEN;
                }
            }
        }
    }
    tokens.push_back(49407); // end of text

    return StatusCode::OJBK;
}

std::string ClipTokenizer::Impl::convert_to_utf8(const std::wstring &input) {

}

std::wstring ClipTokenizer::Impl::convert_to_wstring(const std::string &input) {

}

void ClipTokenizer::Impl::add_special_token(const std::string &token) {

}

/***
 *
 */
ClipTokenizer::ClipTokenizer() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
ClipTokenizer::~ClipTokenizer() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode ClipTokenizer::init(const decltype(toml::parse("")) &cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param input_text
 * @param token
 * @return
 */
StatusCode ClipTokenizer::tokenize(const std::string& input_text, std::vector<int> &token) {
    return _m_pimpl->tokenize(input_text, token);
}

/***
 *
 * @return
 */
bool ClipTokenizer::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

}
}
}