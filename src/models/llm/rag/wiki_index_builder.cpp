/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: WikiPreprocessor.cpp
 * Date: 24-12-9
 ************************************************/

#include "wiki_preprocessor.h"

#include <cctype>
#include <regex>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "fmt/format.h"
#include "rapidjson/writer.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "indicators/indicators.hpp"

#include "common/time_stamp.h"
#include "common/file_path_util.h"
#include "models/llm/llama/llama3.h"

namespace jinq {
namespace models {
namespace llm {

using jinq::common::Timestamp;
using jinq::common::StatusCode;
using jinq::common::FilePathUtil;
using Llama3Ptr = jinq::models::llm::llama::Llama3<std::string, std::string>;

namespace rag {

struct wiki_corpus_segment {
    std::string id;
    std::string url;
    std::string title;
    std::string text;
};

/***************** Impl Function Sets ******************/

class WikiPreprocessor::Impl {
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
    * @param transformer
     */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg
     * @return
     */
    StatusCode init(const decltype(toml::parse("")) &cfg);

    /***
     *
     * @param source_wiki_corpus_dir
     * @param out_save_path
     * @param chunk_word_size
     * @return
     */
    static StatusCode chunk_wiki_corpus(
        const std::string& source_wiki_corpus_dir, const std::string& out_save_path, int chunk_word_size);

    /***
     *
     * @param segmented_corpus_path
     * @param out_index_path
     * @return
     */
    StatusCode build_chunk_index(const std::string& segmented_corpus_path, const std::string& out_index_path);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  private:
    // init flag
    bool _m_successfully_initialized = true;
    // llama3 model
    std::unique_ptr<Llama3Ptr> _m_model;
    // corpus chunk word size
    int32_t _m_chunk_word_size = 100;
    // tokenize max seq length
    int32_t _m_token_max_len = 512;

  private:
    /***
     *
     * @param wiki_str
     * @return
     */
    static StatusCode parse_wiki_records(const std::string& wiki_str, wiki_corpus_segment& out_seg);

    /***
     *
     * @param input
     * @return
     */
    static std::string html_unescape(const std::string& input);

    /***
     *
     * @param segment
     * @return
     */
    static bool preprocess_wiki_corpus(wiki_corpus_segment& segment);

    /***
     *
     * @param str
     * @return
     */
    static bool is_space_or_punct(const std::string& str) {
        auto target = [](const char& c) -> bool {
            if (std::isspace(c) || std::isblank(c) || std::ispunct(c)) {
                return true;
            } else {
                return false;
            }
        };
        auto ret = std::all_of(str.begin(), str.end(), target);
        return ret;
    }
};

/***
 *
 * @param cfg
 * @return
 */
StatusCode WikiPreprocessor::Impl::init(const decltype(toml::parse("")) &cfg) {
    auto wiki_cfg = cfg.at("WIKI_PREPROCESS");

    // init llama3 model
    std::string model_cfg_path = wiki_cfg.at("llm_model_cfg_path").as_string();
    auto model_cfg = toml::parse(model_cfg_path);
    _m_model = std::make_unique<Llama3Ptr>();
    _m_model->init(model_cfg);
    if (!_m_model->is_successfully_initialized()) {
        LOG(ERROR) << "init llama3 model failed";
        _m_successfully_initialized = false;
        return StatusCode::RAG_BUILD_CORPUS_INDEX_FAILED;
    }

    // init preprocess params
    _m_chunk_word_size = static_cast<int32_t >(wiki_cfg["chunk_word_size"].as_integer());
    _m_token_max_len = static_cast<int32_t >(wiki_cfg["tokenize_max_seq_len"].as_integer());

    _m_successfully_initialized = true;
    return StatusCode::OK;
}

/***
 *
 * @param source_wiki_corpus_dir
 * @param out_save_path
 * @param chunk_word_size
 * @return
 */
StatusCode WikiPreprocessor::Impl::chunk_wiki_corpus(
    const std::string &source_wiki_corpus_dir, const std::string &out_save_path, int chunk_word_size) {
    // check if source wiki dir exists
    if (!FilePathUtil::is_dir_exist(source_wiki_corpus_dir)) {
        LOG(ERROR) << fmt::format("source wiki corpus dir: {} not exist", source_wiki_corpus_dir);
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }

    // fetch all wiki corpus paths
    std::vector<std::string> wiki_corpus_paths;
    cv::glob(fmt::format("{}/wiki_*", source_wiki_corpus_dir), wiki_corpus_paths, true);
    if (wiki_corpus_paths.empty()) {
        LOG(ERROR) << fmt::format("source wiki corpus dir: {} empty", source_wiki_corpus_dir);
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }

    // prepare progress bar
    auto progress_bar = std::make_unique<indicators::BlockProgressBar>();
    progress_bar->set_option(indicators::option::BarWidth{80});
    progress_bar->set_option(indicators::option::Start{"["});
    progress_bar->set_option(indicators::option::End{"]"});
    progress_bar->set_option(indicators::option::ForegroundColor{indicators::Color::white});
    progress_bar->set_option(indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
    progress_bar->set_option(indicators::option::ShowElapsedTime{true});
    progress_bar->set_option(indicators::option::ShowPercentage{true});
    progress_bar->set_option(indicators::option::ShowRemainingTime(true));

    // parse all wiki segments
    std::unordered_map<std::string, std::string> corpus;
    for (int idx = 0; idx < wiki_corpus_paths.size(); ++idx) {
        auto f_path = wiki_corpus_paths[idx];
        std::ifstream f_in(f_path, std::ios::in);
        if (!f_in.is_open() || f_in.bad()) {
            LOG(ERROR) << fmt::format("read wiki corpus file: {} failed", f_path);
            return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
        }
        std::string line;
        while (std::getline(f_in, line)) {
            wiki_corpus_segment segment;
            auto status = parse_wiki_records(line, segment);
//            if (!preprocess_wiki_corpus(segment)) {
//                continue;
//            }
            if (status == StatusCode::OK) {
                if (corpus.find(segment.title) == corpus.end()) {
                    corpus.insert(std::make_pair(segment.title, segment.text));
                } else {
                    corpus[segment.title] += fmt::format(" {}", segment.text);
                }
            }
        }
        progress_bar->set_option(indicators::option::PostfixText{fmt::format("parse wiki corpus: {}", idx)});
        progress_bar->set_progress((static_cast<float>(idx) / static_cast<float>(wiki_corpus_paths.size())) * 100.0f);
    }
    progress_bar->mark_as_completed();
    std::vector<std::string> titles;
    std::vector<std::string> texts;
    for (auto& iter : corpus) {
        titles.push_back(iter.first);
        texts.push_back(iter.second);
    }

    // segment corpus by word size
    std::vector<wiki_corpus_segment> clean_segments;
    for (auto idx = 0; idx < texts.size(); ++idx) {
        auto text = texts[idx];
        auto title = titles[idx];
        std::istringstream iss(text);
        std::vector<std::string> total_words;
        std::string word;
        while (iss >> word) {
            total_words.push_back(word);
        }

        int word_count = 0;
        std::vector<std::string> segments;
        std::vector<std::string> segment_words;
        for (auto& w : total_words) {
            segment_words.push_back(fmt::format("{} ", w));
            if (!is_space_or_punct(w)) {
                word_count++;
            }
            if (word_count == chunk_word_size) {
                word_count = 0;
                std::string tmp_seg;
                for (auto& tmp_w : segment_words) {
                    tmp_seg += tmp_w;
                }
                segment_words.clear();
                segments.push_back(tmp_seg);
            }
        }
        while (word_count != chunk_word_size && word_count != 0) {
            for (auto& w : total_words) {
                segment_words.push_back(fmt::format("{} ", w));
                if (!is_space_or_punct(w)) {
                    word_count++;
                }
                if (word_count == chunk_word_size) {
                    break;
                }
            }
        }
        std::string tmp_seg;
        for (auto& tmp_w : segment_words) {
            tmp_seg += tmp_w;
        }
        segment_words.clear();
        segments.push_back(tmp_seg);

        for (auto& segment : segments) {
            std::replace(segment.begin(), segment.end(), '\t', ' ');
            std::replace(segment.begin(), segment.end(), '\n', ' ');
            wiki_corpus_segment clean_segment;
            clean_segment.title = title;
            clean_segment.text = segment;
            clean_segments.push_back(clean_segment);
        }
        progress_bar->set_progress((static_cast<float>(idx) / static_cast<float>(texts.size())) * 100.0f);
        progress_bar->set_option(indicators::option::PostfixText{fmt::format("segment wiki corpus: {}", idx)});
    }

    // write down segmented corpus
    std::ofstream out_f(out_save_path, std::ios::out);
    for (auto idx = 0; idx < clean_segments.size(); ++idx) {
        auto title = clean_segments[idx].title;
        auto text = clean_segments[idx].text;
        rapidjson::Document doc;
        auto& allocator = doc.GetAllocator();
        doc.SetObject();
        doc.AddMember("id", idx, allocator);
        doc.AddMember("title", rapidjson::Value(title.c_str(), allocator), allocator);
        doc.AddMember("text", rapidjson::Value(text.c_str(), allocator), allocator);
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        out_f << buffer.GetString() << std::endl;
    }
    out_f.close();
    LOG(INFO) << "generate segmented wiki corpus complete";

    return StatusCode::OK;
}

/***
 *
 * @param segmented_corpus_path
 * @param out_index_path
 * @return
 */
StatusCode WikiPreprocessor::Impl::build_chunk_index(const std::string &segmented_corpus_path, const std::string &out_index_path) {
    // load corpus file
    if (!FilePathUtil::is_file_exist(segmented_corpus_path)) {
        LOG(ERROR) << fmt::format("corpus file: {} not exists", segmented_corpus_path);
        return StatusCode::RAG_BUILD_CORPUS_INDEX_FAILED;
    }
    std::ifstream f_in(segmented_corpus_path, std::ios::in);
    if (!f_in.is_open() || f_in.bad()) {
        LOG(ERROR) << fmt::format("read corpus file: {} failed", segmented_corpus_path);
        return StatusCode::RAG_BUILD_CORPUS_INDEX_FAILED;
    }
    std::vector<wiki_corpus_segment> corpus;
    std::string line;
    while(std::getline(f_in, line)) {
        rapidjson::Document doc;
        doc.Parse(line.c_str());
        auto id = doc["id"].GetInt64();
        auto title = doc["title"].GetString();
        auto text = doc["text"].GetString();
        wiki_corpus_segment segment{std::to_string(id), "", title, text};
        corpus.push_back(segment);
    }

    // tokenize segments
    std::vector<std::vector<float> > corpus_embeddings;
    for (auto& segment : corpus) {
        auto& text = segment.text;
        auto fmt_input = fmt::format("passage: {}", text);
        std::vector<std::vector<float> > embs;
        auto status = _m_model->embed_prompt(" hello", embs, "mean",  true, _m_token_max_len);
    }


    return StatusCode::OK;
}

/***
 *
 * @param wiki_str
 * @param out_seg
 * @return
 */
StatusCode WikiPreprocessor::Impl::parse_wiki_records(const std::string &wiki_str, wiki_corpus_segment &out_seg) {
    if (wiki_str.empty()) {
        LOG(ERROR) << "empty wiki records";
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }
    rapidjson::Document doc;
    doc.Parse(wiki_str.c_str());
    if (doc.HasParseError() || !doc.IsObject()) {
        LOG(ERROR) << fmt::format("parse wiki string error, ori str: {}", wiki_str);
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }
    if (!doc.HasMember("id")) {
        LOG(ERROR) << fmt::format("parse wiki string error, ori str: {}, miss \'id\' field", wiki_str);
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }
    out_seg.id = doc["id"].GetString();
    if (!doc.HasMember("url")) {
        LOG(ERROR) << fmt::format("parse wiki string error, ori str: {}, miss \'url\' field", wiki_str);
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }
    out_seg.url = doc["url"].GetString();
    if (!doc.HasMember("title")) {
        LOG(ERROR) << fmt::format("parse wiki string error, ori str: {}, miss \'title\' field", wiki_str);
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }
    out_seg.title = doc["title"].GetString();
    if (!doc.HasMember("text")) {
        LOG(ERROR) << fmt::format("parse wiki string error, ori str: {}, miss \'text\' field", wiki_str);
        return StatusCode::RAG_PARSE_WIKI_STR_FAILED;
    }
    out_seg.text = doc["text"].GetString();

    return StatusCode::OK;
}

/***
 *
 * @param input
 * @return
 */
std::string WikiPreprocessor::Impl::html_unescape(const std::string &input) {
    std::string output = input;
    std::regex amp("&amp;");
    std::regex quot("&quot;");
    std::regex lt("&lt;");
    std::regex gt("&gt;");
    std::regex nbsp("&nbsp;");

    output = std::regex_replace(output, amp, "&");
    output = std::regex_replace(output, quot, "\"");
    output = std::regex_replace(output, lt, "<");
    output = std::regex_replace(output, gt, ">");
    output = std::regex_replace(output, nbsp, " ");
    return output;
}

/***
 *
 * @param segment
 * @return
 */
bool WikiPreprocessor::Impl::preprocess_wiki_corpus(wiki_corpus_segment &segment) {
    // html parse
    auto& text = segment.text;
    auto& title = segment.title;
    title = html_unescape(title);
    text = html_unescape(text);

    // remove head and tail
    text.erase(0, text.find_first_not_of(" \t\n\r"));
    text.erase(text.find_last_not_of(" \t\n\r") + 1);

    // check title's validation
    std::string lowerTitle = title;
    std::transform(lowerTitle.begin(), lowerTitle.end(), lowerTitle.begin(), ::tolower);
    if (lowerTitle.find("(disambiguation)") != std::string::npos ||
        lowerTitle.find("(disambiguation page)") != std::string::npos) {
        return false;
    }

    // check list etc.
    std::regex list_pattern(R"((List of .+)|(Index of .+)|(Outline of .+))");
    if (std::regex_match(title, list_pattern)) {
        return false;
    }

    // remove redirect page
    if (text.find("REDIRECT") == 0 || text.find("redirect") == 0) {
        return false;
    }

    // remove references at tail
    if (text.size() > 12 && text.compare(text.size() - 12, 12, ". References.") == 0) {
        text = text.substr(0, text.size() - 12);
        text.erase(text.find_last_not_of(" \t\n\r") + 1);
    }

    std::regex cite_pattern(R"(\{\{cite .*?\}\})");
    text = std::regex_replace(text, cite_pattern, " ");
    text = std::regex_replace(text, std::regex(R"(<math[\s\S]*?</math>)"), "");
    text = std::regex_replace(text, std::regex(R"(<chem[\s\S]*?</chem>)"), "");
    text = std::regex_replace(text, std::regex(R"(<score[\s\S]*?</score>)"), "");

    std::regex table_replace_pattern(R"(TABLETOREPLACE)");
    text = std::regex_replace(text, table_replace_pattern, " ");
    text = std::regex_replace(text, std::regex(R"(\[\[|\]\]|\{\{|\}\}|'''|<br>)"), " ");
    text = std::regex_replace(text, std::regex("&quot;"), "\"");
    text = std::regex_replace(text, std::regex("&amp;"), "&");
    text = std::regex_replace(text, std::regex("nbsp;"), " ");

    text = std::regex_replace(text, std::regex("<.*?/>"), "");

    std::regex filePattern(R"(File:[A-Za-z0-9 ]+\.[a-z]{3,4}(\|[0-9]+px)?)");
    text = std::regex_replace(text, filePattern, "");

    text = std::regex_replace(text, std::regex("[\n\t]"), " ");

    title.erase(std::remove(title.begin(), title.end(), '\n'), title.end());
    title.erase(std::remove(title.begin(), title.end(), '\t'), title.end());

    return true;
}

/************* Export Function Sets *************/

/***
 *
 */
WikiPreprocessor::WikiPreprocessor() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
WikiPreprocessor::~WikiPreprocessor() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode WikiPreprocessor::init(const decltype(toml::parse("")) &cfg) {
   return _m_pimpl->init(cfg);
}

/***
 *
 * @return
 */
bool WikiPreprocessor::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
 *
 * @param source_wiki_corpus_dir
 * @param out_save_path
 * @param chunk_word_size
 * @return
 */
StatusCode WikiPreprocessor::chunk_wiki_corpus(
    const std::string &source_wiki_corpus_dir, const std::string &out_save_path, int chunk_word_size) {
    return _m_pimpl->chunk_wiki_corpus(source_wiki_corpus_dir, out_save_path, chunk_word_size);
}

/***
 *
 * @param segmented_corpus_path
 * @param out_index_path
 * @return
 */
StatusCode WikiPreprocessor::build_chunk_index(const std::string& segmented_corpus_path, const std::string& out_index_path) {
    return _m_pimpl->build_chunk_index(segmented_corpus_path, out_index_path);
}

}
}
}
}
