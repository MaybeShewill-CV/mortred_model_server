/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: WikiIndexBuilder.cpp
 * Date: 24-12-9
 ************************************************/

#include "wiki_index_builder.h"

#include <cctype>
#include <regex>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "fmt/format.h"
#include "rapidjson/writer.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "indicators/indicators.hpp"
#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFFacilities.h"

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

    wiki_corpus_segment() = default;

    ~wiki_corpus_segment() = default;

    wiki_corpus_segment(const wiki_corpus_segment& other) = default;

    wiki_corpus_segment& operator=(const wiki_corpus_segment& other) {
        if (this == &other) {
            return *this;
        }
        id = other.id;
        url = other.url;
        title = other.title;
        text = other.text;
        return *this;
    }

    wiki_corpus_segment(wiki_corpus_segment&& other) noexcept
        : id(std::move(other.id)), url(std::move(other.url)),
          title(std::move(other.title)), text(std::move(other.text)) {}

    wiki_corpus_segment& operator=(wiki_corpus_segment&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        id = std::move(other.id);
        url = std::move(other.url);
        title = std::move(other.title);
        text = std::move(other.text);
        return *this;
    }

};

/***************** Impl Function Sets ******************/

class WikiIndexBuilder::Impl {
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
     * @param out_index_dir
     * @return
     */
    StatusCode build_index(const std::string& source_wiki_corpus_dir, const std::string& out_index_dir);

    /***
     *
     * @param index_file_dir
     * @return
     */
    StatusCode load_index(const std::string& index_file_dir);

    /***
     *
     * @param corpus_segment_dir
     * @return
     */
    StatusCode load_corpus_segment(const std::string& corpus_segment_dir);

    /***
     *
     * @param input_prompt
     * @param out_referenced_corpus
     * @param top_k
     * @param apply_chat_template
     * @return
     */
    StatusCode search(
        const std::string& input_prompt, std::string& out_referenced_corpus, int top_k=1, bool apply_chat_template=true);

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
    std::vector<std::unique_ptr<Llama3Ptr> > _m_encoders;
    // corpus chunk word size
    int32_t _m_chunk_word_size = 100;
    // tokenize max seq length
    int32_t _m_token_max_len = 512;
    // workers
    int _m_segment_workers = 4;

    // index engine
    std::unique_ptr<faiss::IndexFlatL2> _m_index;
    // segmented wiki corpus
    std::vector<wiki_corpus_segment> _m_segment_wiki_corpus;
    // corpus counts
    std::atomic<uint32_t > _m_count_of_source_wiki_corpus{0};
    std::atomic<uint32_t > _m_count_of_source_texts{0};
    std::atomic<uint32_t > _m_count_of_segmented_texts{0};
    std::atomic<uint32_t > _m_count_of_corpus_has_been_parsed{0};
    std::atomic<uint32_t > _m_count_of_texts_has_been_segmented{0};
    std::atomic<uint32_t > _m_count_of_embeddings_has_been_extracted{0};

    struct series_ctx {
        // error code
        std::string err_msg = "success";
        StatusCode err_code = StatusCode::OK;
        // params
        int32_t token_max_len = 256;
        int32_t chunk_word_size = 100;
        int segment_workers = 1;
        int encode_workers = 1;
        std::vector<std::vector<std::string> > split_wiki_corpus_paths;
        std::vector<std::pair<std::vector<std::string>, std::vector<std::string> > > split_wiki_title_texts;
        // output setting
        std::string output_dir;
        std::vector<std::vector<wiki_corpus_segment> > wiki_segments;
        std::vector<std::vector<std::vector<float> > > wiki_segment_features;
    };

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

    /***
     *
     * @param worker_id
     * @param ctx
     * @return
     */
    StatusCode parse_wiki_corpus(int worker_id, series_ctx* ctx);

    /***
     *
     * @param worker_id
     * @param ctx
     * @return
     */
    StatusCode segment_wiki_texts(int worker_id, series_ctx* ctx);

    /***
     *
     * @param ctx
     * @return
     */
    static void resplit_wiki_segments(series_ctx* ctx);

    /***
     *
     * @param worker_id
     * @param ctx
     * @return
     */
    StatusCode embed_segments(int worker_id, series_ctx* ctx);

    /***
     *
     * @param ctx
     * @return
     */
    StatusCode write_index_file(series_ctx* ctx);
};

/***
 *
 * @param cfg
 * @return
 */
StatusCode WikiIndexBuilder::Impl::init(const decltype(toml::parse("")) &cfg) {
    auto wiki_cfg = cfg.at("WIKI_PREPROCESS");

    // init llama3 model
    auto encode_worker_nums = wiki_cfg["encode_worker_nums"].as_integer();
    for (auto i = 0; i < encode_worker_nums; ++i) {
        auto model = std::make_unique<Llama3Ptr>();
        model->init(cfg);
        if (!model->is_successfully_initialized()) {
            LOG(ERROR) << "init llama3 model failed";
            _m_successfully_initialized = false;
            return StatusCode::RAG_BUILD_CORPUS_INDEX_FAILED;
        }
        _m_encoders.push_back(std::move(model));
    }

    // init preprocess params
    _m_chunk_word_size = static_cast<int32_t >(wiki_cfg["chunk_word_size"].as_integer());
    _m_token_max_len = static_cast<int32_t >(wiki_cfg["tokenize_max_seq_len"].as_integer());
    _m_segment_workers = static_cast<int >(wiki_cfg["segment_worker_nums"].as_integer());

    _m_successfully_initialized = true;
    return StatusCode::OK;
}

/***
 *
 * @param source_wiki_corpus_dir
 * @param out_index_dir
 * @return
 */
StatusCode WikiIndexBuilder::Impl::build_index(const std::string &source_wiki_corpus_dir, const std::string &out_index_dir) {
    // init series ctx
    StatusCode status = StatusCode::OK;
    WFFacilities::WaitGroup wait_group(1);
    auto* ctx = new series_ctx;
    ctx->chunk_word_size = _m_chunk_word_size;
    ctx->segment_workers = _m_segment_workers;
    ctx->encode_workers = static_cast<int>(_m_encoders.size());
    ctx->token_max_len = _m_token_max_len;
    ctx->output_dir = out_index_dir;
    ctx->wiki_segments.resize(ctx->segment_workers, std::vector<wiki_corpus_segment>());

    // split source wiki corpus
    std::vector<std::string> src_wiki_corpus_paths;
    cv::glob(fmt::format("{}/wiki_*", source_wiki_corpus_dir), src_wiki_corpus_paths, true);
    if (src_wiki_corpus_paths.empty()) {
        LOG(ERROR) << fmt::format("source wiki corpus dir: {} empty. No wiki_* file found", source_wiki_corpus_dir);
        return StatusCode::RAG_BUILD_CORPUS_INDEX_FAILED;
    }
    _m_count_of_source_wiki_corpus = src_wiki_corpus_paths.size();
    ctx->split_wiki_corpus_paths.resize(ctx->segment_workers, std::vector<std::string>());
    ctx->split_wiki_title_texts.resize(ctx->segment_workers);
    auto split_counts = static_cast<int>(std::ceil(static_cast<float>(src_wiki_corpus_paths.size()) / static_cast<float>(ctx->segment_workers)));
    for (auto i = 0; i < ctx->segment_workers; ++i) {
        for (auto j = i * split_counts; j < (i + 1) * split_counts; ++j) {
            if (j >= src_wiki_corpus_paths.size()) {
                break;
            } else {
                ctx->split_wiki_corpus_paths[i].push_back(src_wiki_corpus_paths[j]);
            }
        }
    }

    // bind task function
    auto&& parse_proc = [&](auto&& PH1, auto&& PH2) {
        return parse_wiki_corpus(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
    };
    auto&& segment_proc = [&](auto&& PH1, auto&& PH2) {
        return segment_wiki_texts(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
    };
    auto&& resplit_proc = [](auto&& PH1) {
        return resplit_wiki_segments(std::forward<decltype(PH1)>(PH1));
    };
    auto&& embed_proc = [this](auto&& PH1, auto&& PH2) {
        return embed_segments(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
    };
    auto&& write_proc = [this](auto&& PH1) {
        return write_index_file(std::forward<decltype(PH1)>(PH1));
    };

    // create task
    auto* resplit_task = WFTaskFactory::create_go_task("resplit_segment_corpus", resplit_proc, ctx);
    auto* write_task = WFTaskFactory::create_go_task("write_down_index", write_proc, ctx);

    // ensemble all tasks
    auto* p_parse_task = Workflow::create_parallel_work(nullptr);
    p_parse_task->set_context(ctx);
    for (auto i = 0; i < ctx->segment_workers; ++i) {
        if (ctx->split_wiki_corpus_paths[i].empty()) {
            continue;
        } else {
            auto worker_id = i;
            auto* gtask = WFTaskFactory::create_go_task("parse_wiki_corpus", parse_proc, worker_id, ctx);
            p_parse_task->add_series(Workflow::create_series_work(gtask, nullptr));
        }
    }
    p_parse_task->set_callback([&](const ParallelWork * pwork) -> void {
        auto* sctx = (series_ctx*)pwork->get_context();
        if (sctx->err_code != StatusCode::OK) {
            status = sctx->err_code;
            return;
        }
        for (auto& iter : sctx->split_wiki_title_texts) {
            _m_count_of_source_texts += iter.first.size();
        }
        std::cout << std::endl;
        LOG(INFO) << fmt::format("parse source wiki corpus complete, total count: {}, parsed: {}",
                                 _m_count_of_source_wiki_corpus, _m_count_of_corpus_has_been_parsed);
        auto* p_segment_task = Workflow::create_parallel_work(nullptr);
        p_segment_task->set_context(ctx);
        for (auto i = 0; i < ctx->segment_workers; ++i) {
            if (ctx->split_wiki_title_texts[i].first.empty()) {
                continue;
            } else {
                auto worker_id = i;
                auto* gtask = WFTaskFactory::create_go_task("segment_wiki_corpus", segment_proc, worker_id, ctx);
                p_segment_task->add_series(Workflow::create_series_work(gtask, nullptr));
            }
        }
        p_segment_task->set_callback([&](const ParallelWork * pwork) -> void {
            auto* sctx = (series_ctx*)pwork->get_context();
            if (sctx->err_code != StatusCode::OK) {
                status = sctx->err_code;
                return;
            }
            for (auto& iter : sctx->wiki_segments) {
                _m_count_of_segmented_texts += iter.size();
            }
            std::cout << std::endl;
            LOG(INFO) << fmt::format(
                "segment wiki titles and texts complete, origin texts count: {}, processed count: {}, segmented texts count: {}",
                _m_count_of_source_texts, _m_count_of_texts_has_been_segmented, _m_count_of_segmented_texts);
            series_of(pwork)->push_back(resplit_task);
        });
        series_of(pwork)->push_back(p_segment_task);
    });

    resplit_task->set_callback([&](const WFGoTask* task) -> void {
        auto* sctx = (series_ctx*)series_of(task)->get_context();
        auto* p_emb_task = Workflow::create_parallel_work(nullptr);
        p_emb_task->set_context(sctx);
        for (auto i = 0; i < ctx->encode_workers; ++i) {
            if (ctx->wiki_segments[i].empty()) {
                continue;
            } else {
                auto worker_id = i;
                auto* gtask = WFTaskFactory::create_go_task("emb_segment_corpus", embed_proc, worker_id, ctx);
                p_emb_task->add_series(Workflow::create_series_work(gtask, nullptr));
            }
        }
        p_emb_task->set_callback([&](const ParallelWork * pwork) -> void {
            std::cout << std::endl;
            LOG(INFO) << fmt::format(
                "extract embedding of segmented texts complete, segmented texts count: {}, extracted count: {}",
                _m_count_of_segmented_texts, _m_count_of_embeddings_has_been_extracted);
            series_of(pwork)->push_back(write_task);
        });
        series_of(task)->push_back(p_emb_task);
    });
    write_task->set_callback([&](const WFGoTask* task) -> void { });

    auto* series = Workflow::create_series_work(p_parse_task, nullptr);
    series->set_context(ctx);
    series->set_callback([&](const SeriesWork * swork) -> void {
        auto* sctx = (series_ctx*)swork->get_context();
        status = sctx->err_code;
        delete sctx;
        wait_group.done();
    });

    series->start();
    wait_group.wait();

    return status;
}

/***
 *
 * @param index_file_dir
 * @return
 */
StatusCode WikiIndexBuilder::Impl::load_index(const std::string &index_file_dir) {
    // gather all *.index file
    std::vector<std::string> index_file_paths;
    cv::glob(fmt::format("{}/*.index", index_file_dir), index_file_paths, true);
    if (index_file_paths.empty()) {
        LOG(ERROR) << fmt::format("no *.index index file found in {}", index_file_dir);
        return StatusCode::RAG_LOAD_INDEX_FAILED;
    }

    // sort index file
    std::sort(index_file_paths.begin(), index_file_paths.end(), [](const std::string& path_a, const std::string& path_b) -> bool {
        auto name_a = FilePathUtil::get_file_name(path_a);
        auto prefix_id_a = name_a.substr(0, name_a.find(".index"));
        prefix_id_a = prefix_id_a.substr(prefix_id_a.rfind('_') + 1);
        auto name_b = FilePathUtil::get_file_name(path_b);
        auto prefix_id_b = name_b.substr(0, name_b.find(".index"));
        prefix_id_b = prefix_id_b.substr(prefix_id_b.rfind('_') + 1);
        return std::stoi(prefix_id_a) < std::stoi(prefix_id_b);
    });

    // load and merge index file
    std::vector<float> merged_vectors;
    int vec_dims = 0;
    for (auto& f_path : index_file_paths) {
        auto* index = dynamic_cast<faiss::IndexFlatL2*>(faiss::read_index(f_path.c_str(), 0));
        if (index == nullptr) {
            LOG(ERROR) << fmt::format("read index file failed: {}", f_path);
            return StatusCode::RAG_LOAD_INDEX_FAILED;
        }
        const float* raw_data = index->get_xb();
        vec_dims = index->d;
        for (size_t i = 0; i < index->ntotal * index->d; ++i) {
            merged_vectors.push_back(raw_data[i]);
        }
    }

    if (_m_index != nullptr) {
        _m_index.reset();
    }
    _m_index = std::make_unique<faiss::IndexFlatL2>(vec_dims);
    auto merged_index_ntotal = static_cast<long>(merged_vectors.size() / vec_dims);
    _m_index->add(merged_index_ntotal, merged_vectors.data());

    return StatusCode::OK;
}

/***
 *
 * @param corpus_segment_dir
 * @return
 */
StatusCode WikiIndexBuilder::Impl::load_corpus_segment(const std::string &corpus_segment_dir) {
    // gather all *.jsonl file
    std::vector<std::string> corpus_file_paths;
    cv::glob(fmt::format("{}/*.jsonl", corpus_segment_dir), corpus_file_paths, true);
    if (corpus_file_paths.empty()) {
        LOG(ERROR) << fmt::format("no *.jsonl corpus file found in {}", corpus_segment_dir);
        return StatusCode::RAG_LOAD_INDEX_FAILED;
    }

    // sort index file
    std::sort(corpus_file_paths.begin(), corpus_file_paths.end(), [](const std::string& path_a, const std::string& path_b) -> bool {
        auto name_a = FilePathUtil::get_file_name(path_a);
        auto prefix_id_a = name_a.substr(0, name_a.find(".jsonl"));
        prefix_id_a = prefix_id_a.substr(prefix_id_a.rfind('_') + 1);
        auto name_b = FilePathUtil::get_file_name(path_b);
        auto prefix_id_b = name_b.substr(0, name_b.find(".jsonl"));
        prefix_id_b = prefix_id_b.substr(prefix_id_b.rfind('_') + 1);
        return std::stoi(prefix_id_a) < std::stoi(prefix_id_b);
    });

    for (auto& corpus_segment_path : corpus_file_paths) {
        std::ifstream f_in(corpus_segment_path, std::ios::in);
        if (!f_in.is_open() || f_in.bad()) {
            LOG(ERROR) << fmt::format("read segment corpus file: {} failed", corpus_segment_path);
            return StatusCode::RAG_LOAD_SEGMENT_CORPUS_FAILED;
        }

        std::string line;
        while (std::getline(f_in, line)) {
            rapidjson::Document doc;
            doc.Parse(line.c_str());
            wiki_corpus_segment segment;
            segment.id = std::to_string(doc["id"].GetInt64());
            segment.title = doc["title"].GetString();
            segment.text = doc["text"].GetString();
            _m_segment_wiki_corpus.push_back(segment);
        }
    }

    return StatusCode::OK;
}

/***
 *
 * @param input_prompt
 * @param out_referenced_corpus
 * @param top_k
 * @param apply_chat_template
 * @return
 */
StatusCode WikiIndexBuilder::Impl::search(
    const std::string &input_prompt, std::string &out_referenced_corpus, int top_k, bool apply_chat_template) {
    // check validation of index and corpus
    if (_m_segment_wiki_corpus.empty()) {
        LOG(ERROR) << "empty segmented corpus load them first";
        return StatusCode::RAG_SEARCH_SEGMENT_CORPUS_FAILED;
    }
    if (_m_index->ntotal == 0) {
        LOG(ERROR) << "empty index load them first";
        return StatusCode::RAG_SEARCH_SEGMENT_CORPUS_FAILED;
    }

    // encode input prompt
    std::vector<std::vector<float> > input_prompt_feats;
    auto status = _m_encoders[0]->get_embedding(input_prompt, input_prompt_feats, "mean", true, _m_token_max_len, true);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("encode input prompt failed status code: {}", status);
        return StatusCode::RAG_SEARCH_SEGMENT_CORPUS_FAILED;
    }

    // search topk corpus;
    int query_nums = 1;
    std::vector<long> ids(top_k * query_nums);
    std::vector<float> dis(top_k * query_nums);
    _m_index->search(query_nums, input_prompt_feats[0].data(), top_k, dis.data(), ids.data());

    // apply template to corpus
    if (apply_chat_template) {
        for (auto i = 0; i < ids.size(); ++i) {
            auto title = _m_segment_wiki_corpus[ids[i]].title;
            auto text = _m_segment_wiki_corpus[ids[i]].text;
            auto fmt_str = fmt::format("Doc {} (Title: {}) {}\n", i + 1, title, text);
            out_referenced_corpus += fmt_str;
        }

    } else {
        for (auto& i : ids) {
            out_referenced_corpus += fmt::format("title: {}\n text: {}\n",
                                             _m_segment_wiki_corpus[i].title, _m_segment_wiki_corpus[i].text);
        }
    }

    return StatusCode::OK;
}


/***
 *
 * @param wiki_str
 * @param out_seg
 * @return
 */
StatusCode WikiIndexBuilder::Impl::parse_wiki_records(const std::string &wiki_str, wiki_corpus_segment &out_seg) {
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
std::string WikiIndexBuilder::Impl::html_unescape(const std::string &input) {
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
bool WikiIndexBuilder::Impl::preprocess_wiki_corpus(wiki_corpus_segment &segment) {
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

/***
 *
 * @param worker_id
 * @param ctx
 * @return
 */
StatusCode WikiIndexBuilder::Impl::parse_wiki_corpus(int worker_id, series_ctx *ctx) {
    // parse all wiki segments
    auto& corpus_paths = ctx->split_wiki_corpus_paths[worker_id];
//    LOG(INFO) << fmt::format("worker {} start parsing source wiki corpus ...", worker_id);
    std::unordered_map<std::string, std::string> corpus;
    for (auto& f_path : corpus_paths) {
        std::ifstream f_in(f_path, std::ios::in);
        if (!f_in.is_open() || f_in.bad()) {
            ctx->err_msg = fmt::format("read wiki corpus file: {} failed", f_path);
            ctx->err_code = StatusCode::RAG_PARSE_WIKI_STR_FAILED;
            LOG(ERROR) << ctx->err_msg;
            return ctx->err_code;
        }
        std::string line;
        while (std::getline(f_in, line)) {
            wiki_corpus_segment segment;
            auto status = parse_wiki_records(line, segment);
            if (!preprocess_wiki_corpus(segment)) {
                continue;
            }
            if (status == StatusCode::OK) {
                if (corpus.find(segment.title) == corpus.end()) {
                    corpus.insert(std::make_pair(segment.title, segment.text));
                } else {
                    corpus[segment.title] += fmt::format(" {}", segment.text);
                }
            }
        }
        _m_count_of_corpus_has_been_parsed++;
        auto info = fmt::format("parsing source wiki corpus: [{} / {}] ...",
                                _m_count_of_corpus_has_been_parsed, _m_count_of_source_wiki_corpus);
        std::cout << "\r" << info << std::flush;
    }
    std::vector<std::string> titles;
    std::vector<std::string> texts;
    for (auto it = corpus.begin(); it != corpus.end();) {
        titles.emplace_back(it->first);
        texts.emplace_back(it->second);
        it = corpus.erase(it);
    }
    corpus.clear();
    corpus.rehash(0);
    ctx->split_wiki_title_texts[worker_id] = std::make_pair(titles, texts);
    return StatusCode::OK;
}

/***
 *
 * @param worker_id
 * @param ctx
 * @return
 */
StatusCode WikiIndexBuilder::Impl::segment_wiki_texts(int worker_id, series_ctx *ctx) {
    auto& titles = ctx->split_wiki_title_texts[worker_id].first;
    auto& texts = ctx->split_wiki_title_texts[worker_id].second;
    if (titles.empty() || texts.empty() || titles.size() != texts.size()) {
        ctx->err_msg = fmt::format("invalid segmented title size: {}, texts size: {}", titles.size(), texts.size());
        ctx->err_code = StatusCode::RAG_PARSE_WIKI_STR_FAILED;
        LOG(ERROR) << ctx->err_msg;
        return ctx->err_code;
    }
    auto& clean_segments = ctx->wiki_segments[worker_id];
    for(auto i = 0; i < titles.size(); ++i) {
        auto& title = titles[i];
        auto& text = texts[i];
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
            if (word_count == ctx->chunk_word_size) {
                word_count = 0;
                std::string tmp_seg;
                for (auto& tmp_w : segment_words) {
                    tmp_seg += tmp_w;
                }
                segment_words.clear();
                segments.push_back(tmp_seg);
            }
        }
        while (word_count != ctx->chunk_word_size && word_count != 0) {
            for (auto& w : total_words) {
                segment_words.push_back(fmt::format("{} ", w));
                if (!is_space_or_punct(w)) {
                    word_count++;
                }
                if (word_count == ctx->chunk_word_size) {
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
        title.clear();
        text.clear();
        _m_count_of_texts_has_been_segmented++;
        auto info = fmt::format("segment source wiki texts: [{} / {}] ...",
                                _m_count_of_texts_has_been_segmented, _m_count_of_source_texts);
        std::cout << "\r" << info << std::flush;
    }

    // write down segmented corpus
    std::string output_f_name = fmt::format("segment_wiki_corpus_{}.jsonl", worker_id);
    std::string out_save_path = FilePathUtil::concat_path(ctx->output_dir, output_f_name);
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

    return StatusCode::OK;
}


/***
 *
 * @param ctx
 * @return
 */
void WikiIndexBuilder::Impl::resplit_wiki_segments(series_ctx *ctx) {
    // gather all segments
    std::vector<wiki_corpus_segment> corpus;
    for (auto& cor : ctx->wiki_segments) {
        for (auto& seg : cor) {
            corpus.push_back(std::move(seg));
        }
    }
    ctx->wiki_segments.clear();
    std::vector<std::vector<wiki_corpus_segment>>().swap(ctx->wiki_segments);

    // resplit segment corpus according to encoder nums
    auto total_corpus_count = corpus.size();
    auto corpus_iter = corpus.begin();
    ctx->wiki_segments.resize(ctx->encode_workers, std::vector<wiki_corpus_segment>());
    auto split_counts = static_cast<int>(std::ceil(static_cast<float>(corpus.size()) / static_cast<float>(ctx->encode_workers)));
    for (auto i = 0; i < ctx->encode_workers; ++i) {
        for (auto j = i * split_counts; j < (i + 1) * split_counts; ++j) {
            if (j >= total_corpus_count) {
                break;
            } else {
                ctx->wiki_segments[i].push_back(std::move(*corpus_iter));
                corpus_iter++;
            }
        }
    }
    corpus.clear();
    std::vector<wiki_corpus_segment>().swap(corpus);

    // resize features
    ctx->wiki_segment_features.resize(ctx->wiki_segments.size());
}

/***
 *
 * @param ctx
 * @return
 */
StatusCode WikiIndexBuilder::Impl::embed_segments(int worker_id, series_ctx *ctx) {
    // load corpus file
    auto& corpus = ctx->wiki_segments[worker_id];
    if (corpus.empty()) {
        ctx->err_msg = fmt::format("empty segmented wiki corpus");
        ctx->err_code = StatusCode::RAG_BUILD_CORPUS_INDEX_FAILED;
        LOG(ERROR) << ctx->err_msg;
        return ctx->err_code;
    }
//    LOG(INFO) << fmt::format("worker {} start embedding segment corpus total counts: {}", worker_id, corpus.size());

    // embedding segments and write index
    auto& model = _m_encoders[worker_id];
    auto model_stat = model->get_model_stat();
    auto embed_dims = model_stat.embed_dims;
    ctx->wiki_segment_features[worker_id].resize(corpus.size());
    for (auto idx = 0; idx < corpus.size(); ++idx) {
        auto& segment = corpus[idx];
        auto& text = segment.text;
        auto fmt_input = fmt::format("passage: {}", text);
        std::vector<std::vector<float> > embs;
        auto status = model->get_embedding(fmt_input, embs, "mean", true, ctx->token_max_len, true);
        if (status != StatusCode::OK) {
            LOG(WARNING) << fmt::format("embed prompt failed status: {}", status);
            std::vector<float> feature(embed_dims, 0.0f);
            ctx->wiki_segment_features[worker_id][idx] = feature;
        } else {
            auto feature = embs[0];
            ctx->wiki_segment_features[worker_id][idx] = feature;
        }
        _m_count_of_embeddings_has_been_extracted++;
        auto info = fmt::format("extract embeddings of segmented texts: [{} / {}] ...",
                                _m_count_of_embeddings_has_been_extracted, _m_count_of_segmented_texts);
        std::cout << "\r" << info << std::flush;
    }
    corpus.clear();
    std::vector<wiki_corpus_segment>().swap(corpus);
//    LOG(INFO) << fmt::format("worker {} extract embedding of wiki corpus complete", worker_id);

    return StatusCode::OK;
}

/***
 *
 * @param ctx
 * @return
 */
StatusCode WikiIndexBuilder::Impl::write_index_file(series_ctx *ctx) {
    size_t total_features_count = 0;
    for (auto& feats : ctx->wiki_segment_features) {
        total_features_count += feats.size();
    }
    if (total_features_count == 0) {
        LOG(ERROR) << "empty segment corpus embedding features";
        return StatusCode::RAG_BUILD_CORPUS_INDEX_FAILED;
    }

    auto model_stat = _m_encoders[0]->get_model_stat();
    auto emb_dims = model_stat.embed_dims;
    LOG(INFO) << "start writing index file ...";
    LOG(INFO) << fmt::format("total segment corpus feature counts: {}", total_features_count);
    LOG(INFO) << fmt::format("feature dims: {}", emb_dims);
    faiss::IndexFlatL2 index(emb_dims);
    for (auto& features : ctx->wiki_segment_features) {
        for (auto& feat : features) {
            index.add(1, feat.data());
            feat.clear();
        }
        features.clear();
        std::vector<std::vector<float> >().swap(features);
    }

    std::string output_name = "wiki_index_flatl2.index";
    auto output_path = FilePathUtil::concat_path(ctx->output_dir, output_name);
    faiss::write_index(&index, output_path.c_str(), 0);
    LOG(INFO) << fmt::format("successfully write down index file into: {}", output_path);

    return StatusCode::OJBK;
}

/************* Export Function Sets *************/

/***
 *
 */
WikiIndexBuilder::WikiIndexBuilder() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 */
WikiIndexBuilder::~WikiIndexBuilder() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode WikiIndexBuilder::init(const decltype(toml::parse("")) &cfg) {
   return _m_pimpl->init(cfg);
}

/***
 *
 * @return
 */
bool WikiIndexBuilder::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
 *
 * @param source_wiki_corpus_dir
 * @param out_index_dir
 * @return
 */
StatusCode WikiIndexBuilder::build_index(const std::string& source_wiki_corpus_dir, const std::string& out_index_dir) {
    return _m_pimpl->build_index(source_wiki_corpus_dir, out_index_dir);
}

/***
 *
 * @param index_file_dir
 * @return
 */
StatusCode WikiIndexBuilder::load_index(const std::string &index_file_dir) {
    return _m_pimpl-> load_index(index_file_dir);
}

/***
 *
 * @param corpus_segment_dir
 * @return
 */
StatusCode WikiIndexBuilder::load_corpus_segment(const std::string &corpus_segment_dir) {
    return _m_pimpl-> load_corpus_segment(corpus_segment_dir);
}

/***
 *
 * @param input_prompt
 * @param out_referenced_corpus
 * @param top_k
 * @param apply_chat_template
 * @return
 */
StatusCode WikiIndexBuilder::search(
    const std::string &input_prompt, std::string &out_referenced_corpus, int top_k, bool apply_chat_template) {
    return _m_pimpl->search(input_prompt, out_referenced_corpus, top_k, apply_chat_template);
}

}
}
}
}
