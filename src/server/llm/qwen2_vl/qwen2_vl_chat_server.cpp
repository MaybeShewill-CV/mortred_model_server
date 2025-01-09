/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: Qwen2VLChatServer.cpp
 * Date: 25-1-8
 ************************************************/

#include "qwen2_vl_chat_server.h"

#include <random>
#include <sstream>
#include <iomanip>

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "fmt/format.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"
#include "workflow/Workflow.h"
#include "workflow/HttpUtil.h"
#include "workflow/HttpMessage.h"

#include "common/status_code.h"
#include "common/file_path_util.h"
#include "models/model_io_define.h"
#include "models/llm/llm_datatype.hpp"
#include "models/llm/qwen2_vl/qwen2_vl.h"

namespace jinq {
namespace server {

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;

namespace llm {

using models::llm::Dialog;
using models::llm::ChatMessage;
using models::io_define::llm::vlm::bytes_input;
using models::io_define::llm::vlm::std_vlm_output;
using ModelPtr = models::llm::qwen2_vl::Qwen2VL<bytes_input, std_vlm_output>;

namespace qwen2_vl {

namespace server_internal_impl {

/***
 *
 * @return
 */
std::string generate_uuid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream uuid;
    uuid << std::hex << std::setfill('0');

    for (int i = 0; i < 8; ++i) {
        uuid << dis(gen);
    }
    uuid << "-";
    for (int i = 0; i < 4; ++i) {
        uuid << dis(gen);
    }
    uuid << "-4";  //
    for (int i = 0; i < 3; ++i) {
        uuid << dis(gen);
    }
    uuid << "-";
    for (int i = 0; i < 4; ++i) {
        uuid << dis(gen);
    }
    uuid << "-";
    for (int i = 0; i < 12; ++i) {
        uuid << dis(gen);
    }

    return uuid.str();
}
}

/************ Impl Declaration ************/

class Qwen2VLChatServer::Impl {
  public:
    /***
    *
    * @param cfg_file_path
    * @return
     */
    StatusCode init(const decltype(toml::parse("")) &config);

    /***
     *
     * @param task
     */
    void serve_process(WFHttpTask* task);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

  public:
    // server params
    int max_connection_nums = 200;
    int peer_resp_timeout = 15 * 1000;
    int compute_threads = -1;
    int handler_threads = 25;
    size_t request_size_limit = -1;

  private:
    // init flag
    bool _m_successfully_initialized = false;
    // task count
    std::atomic<size_t> _m_received_jobs{0};
    std::atomic<size_t> _m_finished_jobs{0};
    std::atomic<size_t> _m_waiting_jobs{0};
    // model run timeout
    int _m_model_run_timeout = 500; // ms
    // server uri
    std::string _m_server_uri;
    // llama3 generator
    std::unique_ptr<ModelPtr > _m_generator;

  private:
    // dialog task
    struct dialog_task {
        std::string task_id;
        bool is_valid = true;
        std::string uuid;
        Dialog current_dialog;
    };
    // workflow series context
    struct seriex_ctx {
        protocol::HttpResponse* response = nullptr;
        StatusCode err_state = StatusCode::OK;
        std::string err_msg = "success";
        std::string task_id;
        std::string task_received_ts;
        std::string task_finished_ts;
        bool is_task_req_valid = false;
        std::string gen_out;
        dialog_task* d_task = nullptr;
    };
    // dialog cache
    std::unordered_map<std::string, Dialog> _m_user_history_dialogs;

  private:
    /***
     *
     * @param req
     * @param task
     * @return
     */
    static StatusCode parse_request(const protocol::HttpRequest* req, dialog_task* task);

    /***
     *
     * @param task
     * @param ctx
     */
    void complete_chat(seriex_ctx* ctx);

    /***
     *
     * @param task
     */
    void complete_chat_cb(const WFGoTask* task);

    /***
     *
     * @param ctx
     * @param dropped_token_ratio
     * @param max_summary_token_ratio
     * @return
     */
    StatusCode regenerate_with_cache_dialogs(seriex_ctx* ctx, float dropped_token_ratio=0.5, float max_summary_token_ratio=0.1);
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode Qwen2VLChatServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    if (!config.contains("QWEN2_VL_CHAT_SERVER")) {
        LOG(ERROR) << (fmt::format(R"(config file doesn't contain filed: "QWEN2_VL_CHAT_SERVER")"));
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto server_section = config.at("QWEN2_VL_CHAT_SERVER");
    auto model_section = config.at("QWEN2_VL_CHAT_MODEL");
    std::string model_cfg_path = model_section.at("model_config_file_path").as_string();
    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << (fmt::format("model config file: {} not exist", model_cfg_path));
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg = toml::parse(model_cfg_path);
    _m_generator = std::make_unique<ModelPtr>();
    auto status = _m_generator->init(model_cfg);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("init qwen2-vl model failed, status code: {}", std::to_string(status));
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    // init server uri
    if (!server_section.contains("server_url")) {
        LOG(ERROR) << "missing server uri field";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    } else {
        _m_server_uri = server_section.at("server_url").as_string();
    }

    // init server params
    max_connection_nums = static_cast<int>(server_section.at("max_connections").as_integer());
    peer_resp_timeout = static_cast<int>(server_section.at("peer_resp_timeout").as_integer()) * 1000;
    compute_threads = static_cast<int>(server_section.at("compute_threads").as_integer());
    handler_threads = static_cast<int>(server_section.at("handler_threads").as_integer());
    request_size_limit = static_cast<size_t>(server_section.at("request_size_limit").as_integer());

    _m_successfully_initialized = true;
    LOG(INFO) << "qwen2-vl chat server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void Qwen2VLChatServer::Impl::serve_process(WFHttpTask* task) {
    // welcome message
    if (strcmp(task->get_req()->get_request_uri(), "/welcome") == 0) {
        task->get_resp()->append_output_body("<html>Welcome to jinq ai server</html>");
        return;
    }
    // hello world message
    else if (strcmp(task->get_req()->get_request_uri(), "/hello_world") == 0) {
        task->get_resp()->append_output_body("<html>Hello World !!!</html>");
        return;
    }
    // check model stat
    else if (strcmp(task->get_req()->get_request_uri(), "/check_model_stat") == 0) {
        auto model_stat = _m_generator->get_model_stat();
        task->get_resp()->append_output_body(fmt::format(
            "<html>n_ctx: {}\n kv cache used: {}\n clip_embedding_dims: {}\n clip_hidden_size: {} \n</html>",
            model_stat.n_ctx_size, model_stat.kv_cache_cell_nums, model_stat.clip_embedding_dims, model_stat.clip_hidden_size));
        return;
    }
    // clear kv cache
    else if (strcmp(task->get_req()->get_request_uri(), "/clear_kv_cache") == 0) {
        _m_generator->clear_kv_cache_cell();
        auto model_stat = _m_generator->get_model_stat();
        task->get_resp()->append_output_body(fmt::format(
            "<html>n_ctx: {}\n kv cache used: {}\n clip_embedding_dims: {}\n clip_hidden_size: {} \n</html>",
            model_stat.n_ctx_size, model_stat.kv_cache_cell_nums, model_stat.clip_embedding_dims, model_stat.clip_hidden_size));
        return;
    }
    // clear kv cache
    else if (strcmp(task->get_req()->get_request_uri(), "/get_context_perf") == 0) {
        auto data = _m_generator->get_context_perf();
        const double t_end_ms = 1e-3 * static_cast<double>(ggml_time_us());
        auto perf_str = fmt::format(
            "load time = {} ms\n"
            "prompt eval time = {} ms / %5d tokens ({} ms per token, {} tokens per second)\n"
            "eval time = {} ms / {} runs   ({} ms per token, {} tokens per second)\n"
            "total time = {} ms / {} tokens\n",
            data.t_load_ms,
            data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval,
            data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval,
            (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval)
        );
        task->get_resp()->append_output_body(fmt::format(
            "<html>context perf data: {}</html>", perf_str));
        return;
    }
    // model service
    else if (strcmp(task->get_req()->get_request_uri(), _m_server_uri.c_str()) == 0) {
        // parse request body
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto* d_task = new dialog_task;
        parse_request(req, d_task);
        if (!d_task->is_valid) {
            task->get_resp()->append_output_body(fmt::format("invalid request data: {}", protocol::HttpUtil::decode_chunked_body(req)));
            return;
        }
        _m_waiting_jobs++;
        _m_received_jobs++;

        // init series work
        auto* series = series_of(task);
        auto* ctx = new seriex_ctx;
        ctx->response = resp;
        ctx->d_task = d_task;
        series->set_context(ctx);
        series->set_callback([&](const SeriesWork * s_work) -> void {
            auto* s_ctx = (seriex_ctx*)s_work->get_context();
            delete s_ctx;
        });

        auto&& go_proc = [this](auto&& PH1) {
            complete_chat(std::forward<decltype(PH1)>(PH1));
        };
        auto* go_task = WFTaskFactory::create_go_task(_m_server_uri, go_proc, ctx);
        auto&& go_proc_cb = [this](auto&& PH1) {
            complete_chat_cb(std::forward<decltype(PH1)>(PH1));
        };
        go_task->set_callback(go_proc_cb);

        *series << go_task;
        return;
    }
}

/***
 *
 * @param req
 * @param task
 * @return
 */
StatusCode Qwen2VLChatServer::Impl::parse_request(const protocol::HttpRequest* req, dialog_task* task) {
    // set task uuid
    protocol::HttpHeaderMap map(req);
    if (!map.key_exists("cookie")) {
        task->uuid = server_internal_impl::generate_uuid();
    } else {
        task->uuid = map.get("cookie");
    }

    std::string req_body = protocol::HttpUtil::decode_chunked_body(req);
    rapidjson::Document doc;
    doc.Parse(req_body.c_str());
    if (!doc.IsObject()) {
        task->is_valid = false;
        LOG(ERROR) << fmt::format("parse request body failed, invalid json str: {}", req_body);
        return StatusCode::SERVER_RUN_FAILED;
    }

    if (doc.HasMember("task_id")) {
        task->task_id = doc["task_id"].GetString();
    }

    if (!doc.HasMember("data")) {
        task->is_valid = false;
        LOG(ERROR) << (fmt::format("invalid json str: {}, missing \"data\" field", req_body));
        return StatusCode::SERVER_RUN_FAILED;
    }
    auto messages = doc["data"].GetArray();
    for (auto& msg : messages) {
        std::string role = msg["role"].GetString();
        std::string content;
        if (role == "user") {
            rapidjson::Document tmp_doc;
            tmp_doc.SetObject();
            tmp_doc.AddMember("content", msg["content"].GetArray(), tmp_doc.GetAllocator());
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            tmp_doc.Accept(writer);
            content = buffer.GetString();
        } else {
            content = msg["content"].GetString();
        }
        task->current_dialog.push_back({role, content});
    }

    return StatusCode::OK;
}

/***
 *
 * @param task
 * @param ctx
 */
void Qwen2VLChatServer::Impl::complete_chat(seriex_ctx* ctx) {
    // fetch current dialog
    auto task = ctx->d_task;
    Dialog dialog = task->current_dialog;

    // generate response
    auto status = _m_generator->chat_completion(task->current_dialog, ctx->gen_out);
    if (status == StatusCode::VLM_QWEN_PARSE_IMAGE_URL_FAILED) {
        ctx->err_state = status;
        ctx->err_msg = "fetch image bytes data from url failed, plz check if url exists or valid";
        return;
    }

    // cache history dialog
    ChatMessage msg = {"assistant", ctx->gen_out};
    dialog.push_back(msg);
    if (_m_user_history_dialogs.find(task->uuid) != _m_user_history_dialogs.end()) {
        _m_user_history_dialogs[task->uuid] += dialog;
    } else {
        _m_user_history_dialogs.insert(std::make_pair(task->uuid, dialog));
    }

    // check if context exceeded occurred
    if (status == StatusCode::LLM_CONTEXT_SIZE_EXCEEDED) {
        int try_times = 5;
        float base_token_drop_ration = 0.75f;
        float base_summary_token_ratio = 0.1f;
        float scale_ratio = 1.4f;
        while (try_times--) {
            status = regenerate_with_cache_dialogs(ctx, base_token_drop_ration, base_summary_token_ratio);
            if (status == StatusCode::LLM_CONTEXT_SIZE_EXCEEDED) {
                LOG(WARNING) << "context still exceeded during regeneration process, try another time with more token dropped";
                base_token_drop_ration *= scale_ratio;
                base_summary_token_ratio /= scale_ratio;
                continue;
            } else {
                break;
            }
        }
    }

    // fill in ctx messages
    if (status == StatusCode::OK) {
        return;
    } else {
        auto err_msg = fmt::format("complete chat failed, status: {}", std::to_string(status));
        ctx->err_msg = err_msg;
        ctx->err_state = status;
        LOG(ERROR) << (err_msg);
        return;
    }
}

/***
 *
 * @param g_task
 */
void Qwen2VLChatServer::Impl::complete_chat_cb(const WFGoTask* task) {
    auto state = task->get_state();
    auto error = task->get_error();
    auto* ctx = (seriex_ctx*)series_of(task)->get_context();

    // fill response
    if (state != WFT_STATE_SUCCESS) {
        ctx->err_state = StatusCode::SERVER_RUN_FAILED;
        ctx->err_msg = fmt::format("workflow go task exec failed, state: {}, msg: {}", state, WFGlobal::get_error_string(state, error));
        LOG(ERROR) << ctx->err_msg;
    }

    std::string task_id = ctx->is_task_req_valid ? ctx->task_id : "";
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
    doc.AddMember("code", ctx->err_state, allocator);
    doc.AddMember("msg", rapidjson::Value(ctx->err_msg.c_str(), allocator), allocator);
    rapidjson::Value data;
    data.SetObject();
    data.AddMember("task_id",  rapidjson::Value(task_id.c_str(), allocator), allocator);
    data.AddMember("response",  rapidjson::Value(ctx->gen_out.c_str(), allocator), allocator);
    doc.AddMember("data", data, allocator);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    auto response_body = buffer.GetString();
    ctx->response->append_output_body(response_body);
    ctx->response->add_header_pair("Set-Cookie", ctx->d_task->uuid);

    // update task count
    _m_finished_jobs++;
    _m_waiting_jobs--;
}

/***
 *
 * @param ctx
 * @param dropped_token_ratio
 * @param max_summary_token_ratio
 * @return
 */
StatusCode Qwen2VLChatServer::Impl::regenerate_with_cache_dialogs(
    seriex_ctx *ctx, float dropped_token_ratio, float max_summary_token_ratio) {
//    auto task = ctx->d_task;
//    // prepare summary dialog
//    Dialog summary_dialogs;
//    auto history_dialogs = _m_user_history_dialogs[task->uuid];
//    std::string fmt_string;
//    auto status = _m_generator->apply_chat_template(history_dialogs, false, fmt_string);
//    if (status != StatusCode::OK) {
//        return status;
//    }
//    std::vector<llama_token > fmt_tokens;
//    status = _m_generator->tokenize(fmt_string, fmt_tokens, true);
//    if (status != StatusCode::OK) {
//        return status;
//    }
//    auto history_dialog_tokens = fmt_tokens.size();
//    auto drop_threshold = static_cast<int32_t >(dropped_token_ratio * static_cast<float>(history_dialog_tokens));
//    size_t dropped_token_nums = 0;
//    int msg_idx =0;
//    for (; msg_idx < history_dialogs.size(); ++msg_idx) {
//        auto role = history_dialogs[msg_idx].role;
//        auto content = history_dialogs[msg_idx].content;
//        Dialog tmp_dia(role, content);
//        _m_generator->apply_chat_template(tmp_dia, false, fmt_string);
//        _m_generator->tokenize(fmt_string, fmt_tokens, true);
//        dropped_token_nums += fmt_tokens.size();
//        summary_dialogs += tmp_dia;
//        if (dropped_token_nums >= drop_threshold) {
//            msg_idx++;
//            break;
//        }
//    }
//    auto summary_token_nums = static_cast<int32_t >(static_cast<float>(dropped_token_nums) * max_summary_token_ratio);
//    summary_token_nums = summary_token_nums > 0 ? summary_token_nums : 1;
//    summary_dialogs.push_back({"system", "You are an assistant skilled at generating summaries."});
//    summary_dialogs.push_back(
//        {"user", fmt::format("Please summarize the multi-turn conversation above "
//                             "in content not exceeding {} tokens.", summary_token_nums)}
//    );
//
//    // check summary dialog token nums
//    _m_generator->apply_chat_template(summary_dialogs, false, fmt_string);
//    _m_generator->tokenize(fmt_string, fmt_tokens, true);
//    auto summary_tokens = static_cast<double>(fmt_tokens.size());
//    auto n_ctx = _m_generator->get_model_stat().n_ctx_size;
//    while (summary_tokens > 0.75 * n_ctx) {
//        summary_dialogs.messages.erase(summary_dialogs.messages.begin());
//        _m_generator->apply_chat_template(summary_dialogs, false, fmt_string);
//        _m_generator->tokenize(fmt_string, fmt_tokens, true);
//        summary_tokens = static_cast<double>(fmt_tokens.size());
//    }
//    LOG(INFO) << "n_tokens: " << summary_tokens << " used before summary";
//
//    // generate summary msg
//    _m_generator->clear_kv_cache_cell();
//    std::string summary_msg;
//    status = _m_generator->chat_completion(summary_dialogs, summary_msg);
//    if (status != StatusCode::OK) {
//        return status;
//    }
//    LOG(INFO) << "summary msg: " << summary_msg;
//
//    // renew history dialogs
//    Dialog updated_dialog(
//        "system",
//        fmt::format("You are a smart ai assistant from Mortred Company.Here is the summary of our previous {} rounds of "
//                    "conversation. Summary content is {}.Please continue assisting the customer based on it.",
//                    summary_dialogs.size(), summary_msg)
//    );
//    _m_generator->apply_chat_template(updated_dialog, false, fmt_string);
//    _m_generator->tokenize(fmt_string, fmt_tokens, true);
//    LOG(INFO) << "n_tokens: " << fmt_tokens.size() << " used after summary";
//    for (auto i = msg_idx; i < history_dialogs.size(); ++i) {
//        updated_dialog.push_back(history_dialogs[i]);
//    }
//    _m_user_history_dialogs[task->uuid].clean_cache();
//    _m_user_history_dialogs[task->uuid] = updated_dialog;
//
//    // regenerate response content
//    _m_generator->clear_kv_cache_cell();
//    Dialog cur_dialog = updated_dialog + task->current_dialog;
//    status = _m_generator->chat_completion(cur_dialog, ctx->gen_out);
//
//    // cache dialog
//    _m_user_history_dialogs[task->uuid] += task->current_dialog;
//    _m_user_history_dialogs[task->uuid] += Dialog("assistant", ctx->gen_out);

    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
 *
 */
Qwen2VLChatServer::Qwen2VLChatServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
Qwen2VLChatServer::~Qwen2VLChatServer() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode Qwen2VLChatServer::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);
    if (status != StatusCode::OK) {
        LOG(INFO) << "init llama3 chat server failed";
        return status;
    }

    // init server
    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.compute_threads = _m_impl->compute_threads;
    settings.handler_threads = _m_impl->handler_threads;
    WORKFLOW_library_init(&settings);

    WFServerParams server_params = SERVER_PARAMS_DEFAULT;
    server_params.max_connections = _m_impl->max_connection_nums;
    server_params.peer_response_timeout = _m_impl->peer_resp_timeout;
    server_params.request_size_limit = _m_impl->request_size_limit * 1024 * 1024;

    auto&& proc = [&](auto arg) {
        return this->_m_impl->serve_process(arg);
    };
    _m_server = std::make_unique<WFHttpServer>(&server_params, proc);

    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void Qwen2VLChatServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool Qwen2VLChatServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}

}
}
}
}
