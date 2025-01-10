/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llam3_chat_server.cpp
 * Date: 24-11-29
 ************************************************/

#include "llama3_chat_server.h"

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
#include "models/llm/llm_datatype.hpp"
#include "models/llm/llama/llama3.h"

namespace jinq {
namespace server {

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;

namespace llm {

using models::llm::Dialog;
using models::llm::ChatMessage;
using LLMPtr = models::llm::llama::Llama3<std::vector<llama_token>&, std::string>;

namespace llama {

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

class Llama3ChatServer::Impl {
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
    std::unique_ptr<LLMPtr > _m_generator;

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
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode Llama3ChatServer::Impl::init(const decltype(toml::parse("")) &config) {
    // init working queue
    if (!config.contains("LLAMA3_CHAT_SERVER")) {
        LOG(ERROR) << (fmt::format(R"(config file doesn't contain filed: "LLAMA3_CHAT_SERVER")"));
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto server_section = config.at("LLAMA3_CHAT_SERVER");
    auto model_section = config.at("LLAMA3_CHAT_MODEL");
    std::string model_cfg_path = model_section.at("model_config_file_path").as_string();
    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << (fmt::format("model config file: {} not exist", model_cfg_path));
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg = toml::parse(model_cfg_path);
    _m_generator = std::make_unique<LLMPtr>();
    auto status = _m_generator->init(model_cfg);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("init llama3 model failed, status code: {}", std::to_string(status));
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
    LOG(INFO) << "llama3 chat server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param task
 */
void Llama3ChatServer::Impl::serve_process(WFHttpTask* task) {
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
            "<html>n_ctx: {}\n kv cache used: {}</html>", model_stat.n_ctx_size, model_stat.kv_cache_cell_nums));
        return;
    }
    // clear kv cache
    else if (strcmp(task->get_req()->get_request_uri(), "/clear_kv_cache") == 0) {
        _m_generator->clear_kv_cache_cell();
        auto model_stat = _m_generator->get_model_stat();
        task->get_resp()->append_output_body(fmt::format(
            "<html>kv cache cleared.\n n_ctx: {}\n kv cache used: {}</html>", model_stat.n_ctx_size, model_stat.kv_cache_cell_nums));
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
StatusCode Llama3ChatServer::Impl::parse_request(const protocol::HttpRequest* req, dialog_task* task) {
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
        std::string content = msg["content"].GetString();
        task->current_dialog.push_back({role, content});
    }

    return StatusCode::OK;
}

/***
 *
 * @param task
 * @param ctx
 */
void Llama3ChatServer::Impl::complete_chat(seriex_ctx* ctx) {
    // fetch current dialog
    auto task = ctx->d_task;
    Dialog dialog = task->current_dialog;

    // generate response
    auto status = _m_generator->chat_completion(task->current_dialog, ctx->gen_out);

    // check if context exceeded occurred
    if (status == StatusCode::LLM_CONTEXT_SIZE_EXCEEDED) {
        auto model_stat = _m_generator->get_model_stat();
        LOG(INFO) << fmt::format("context size: {}", model_stat.n_ctx_size);
        LOG(INFO) << fmt::format("kv cache used cell counts: {} before shift", model_stat.kv_cache_cell_nums);
        LOG(INFO) << fmt::format("kv cache token counts: {} before shift", model_stat.kv_cache_token_nums);

        // shift kv cache
        const int n_keep = 1; // begin of the text token(bos token)
        const int n_left = model_stat.kv_cache_cell_nums - n_keep;
        const int n_discard = n_left / 2;
        LOG(INFO) << fmt::format("context shift, n_keep = {}, n_left = {}, n_discard = {}", n_keep, n_left, n_discard);
        int try_times = 5;
        while (try_times--) {
            status = _m_generator->kv_cache_shift_top_n(n_discard, 0);
            if (status == StatusCode::OK) {
                break;
            }
        }
        if (try_times < 0) {
            LOG(ERROR) << "shift kv cache failed, clear kv cache";
            _m_generator->clear_kv_cache_cell();
        }
        model_stat = _m_generator->get_model_stat();
        LOG(INFO) << fmt::format("kv cache used cell counts: {} after shift", model_stat.kv_cache_cell_nums);
        LOG(INFO) << fmt::format("kv cache token counts: {} after shift", model_stat.kv_cache_token_nums);

        // re-generate response
        status = _m_generator->chat_completion(task->current_dialog, ctx->gen_out);
    }

    // cache history dialog
    ChatMessage msg = {"assistant", ctx->gen_out};
    dialog.push_back(msg);
    if (_m_user_history_dialogs.find(task->uuid) != _m_user_history_dialogs.end()) {
        _m_user_history_dialogs[task->uuid] += dialog;
    } else {
        _m_user_history_dialogs.insert(std::make_pair(task->uuid, dialog));
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
void Llama3ChatServer::Impl::complete_chat_cb(const WFGoTask* task) {
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

/************* Export Function Sets *************/

/***
 *
 */
Llama3ChatServer::Llama3ChatServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
Llama3ChatServer::~Llama3ChatServer() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode Llama3ChatServer::init(const decltype(toml::parse("")) &config) {
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
void Llama3ChatServer::serve_process(WFHttpTask* task) {
    return _m_impl->serve_process(task);
}

/***
 *
 * @return
 */
bool Llama3ChatServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}

}
}
}
}
