/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: llam3_chat_server.cpp
 * Date: 24-11-29
 ************************************************/

#include "llama3_chat_server.h"

#include <iomanip>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>

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
#include "common/llm_request_parser.h"
#include "models/llm/llm_datatype.hpp"
#include "models/llm/llama/llama3.h"
#include "server/base_server_impl.h"

namespace jinq {
namespace server {

using jinq::common::StatusCode;
using jinq::common::FilePathUtil;

namespace llm {

using models::llm::Dialog;
using models::llm::ChatMessage;

namespace llama {

using models::llm::llama::Llama3;

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

using LlamaModelPtr = std::unique_ptr<models::llm::llama::Llama3<std::vector<llama_token>&, std::string> >;

/************ Impl Declaration ************/

class Llama3ChatServer::Impl : public BaseAiServerImpl<LlamaModelPtr, std::string> {
public:
    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const toml::table &config) override;

protected:
    /***
     *
     * @param req
     * @return
     */
    task_request parse_task_request(const protocol::HttpRequest* req) override;

    /***
     *
     * @param req
     * @param ctx
     */
    void do_work(const task_request& req, std::shared_ptr<series_ctx> ctx) override;

    /***
     *
     * @param task_id
     * @param status
     * @param model_output
     * @return
     */
    std::string make_response_body(
        const std::string& task_id,
        const StatusCode& status,
        const std::string& model_output) override;

    /***
     *
     * @param task
     * @return
     */
    bool handle_custom_endpoint(WFHttpTask* task) override;

private:
    // llama3 generator：单 worker，入队 _m_working_queue 实现串行化与会话亲和
    LlamaModelPtr _m_generator;
    // dialog cache
    std::unordered_map<std::string, Dialog> _m_user_history_dialogs;

    /***
     * 状态快照：由持有 worker 的线程（init / do_work / clear_kv_cache）在独占安全点更新，
     * 状态端点只读快照，避免 handler 线程阻塞等待推理 worker。
     */
    struct LlmStatusSnapshot {
        jinq::models::llm::ModelStatus model_status;
        llama_perf_context_data context_perf{};
    };

    mutable std::mutex _m_snapshot_mutex;
    LlmStatusSnapshot _m_status_snapshot;

    /***
     * 在独占 worker 的安全点更新状态快照。
     */
    void update_status_snapshot(const LlamaModelPtr& worker) {
        std::lock_guard<std::mutex> lock(_m_snapshot_mutex);
        _m_status_snapshot.model_status = worker->get_model_stat();
        _m_status_snapshot.context_perf = worker->get_context_perf();
    }

    /***
     * 读取状态快照（handler 线程调用，不触碰 worker 队列）。
     */
    LlmStatusSnapshot read_status_snapshot() const {
        std::lock_guard<std::mutex> lock(_m_snapshot_mutex);
        return _m_status_snapshot;
    }
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode Llama3ChatServer::Impl::init(const toml::table &config) {
    if (!config.contains("LLAMA3_CHAT_SERVER")) {
        LOG(ERROR) << (fmt::format(R"(config file doesn't contain filed: "LLAMA3_CHAT_SERVER")"));
        return StatusCode::SERVER_INIT_FAILED;
    }
    const toml::table* server_section_ptr = config["LLAMA3_CHAT_SERVER"].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section LLAMA3_CHAT_SERVER missing or not a table";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    const toml::table& server_section = *server_section_ptr;

    auto common_status = parse_common_server_config(server_section);
    if (common_status != StatusCode::OK) {
        return common_status;
    }
    auto model_section = config["LLAMA3_CHAT_MODEL"];
    std::string model_cfg_path = model_section["model_config_file_path"].value_or<std::string>("");
    if (!FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << (fmt::format("model config file: {} not exist", model_cfg_path));
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg_parsed = toml::parse_file(model_cfg_path);
    if (!model_cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: " << std::string(model_cfg_parsed.error().description());
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg = std::move(model_cfg_parsed).table();
    _m_generator = std::make_unique<Llama3<std::vector<llama_token>&, std::string> >();
    auto status = _m_generator->init(model_cfg);
    if (status != StatusCode::OK) {
        LOG(ERROR) << fmt::format("init llama3 model failed, status code: {}", std::to_string(status));
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    }

    // 单 worker 入队：所有聊天请求经阻塞队列串行执行，保证 KV cache 会话亲和且无数据竞争
    update_status_snapshot(_m_generator);
    _m_working_queue.enqueue(std::move(_m_generator));

    // init server uri
    if (!server_section.contains("server_url")) {
        LOG(ERROR) << "missing server uri field";
        _m_successfully_initialized = false;
        return StatusCode::SERVER_INIT_FAILED;
    } else {
        _m_server_uri = server_section["server_url"].value_or<std::string>("");
    }

    // 聊天生成时间无界，不设推理超时
    _m_model_run_timeout = -1;
    // 单 handler 线程串行处理 HTTP 请求，保持原有行为
    _m_handler_threads = 1;

    _m_successfully_initialized = true;
    LOG(INFO) << "llama3 chat server init successfully";
    return StatusCode::OK;
}

/***
 *
 * @param req
 * @return
 */
Llama3ChatServer::Impl::task_request Llama3ChatServer::Impl::parse_task_request(const protocol::HttpRequest* req) {
    task_request result;

    // 会话 id：优先复用 cookie，否则生成新 uuid
    protocol::HttpHeaderMap map(req);
    if (!map.key_exists("cookie")) {
        result.session_id = server_internal_impl::generate_uuid();
    } else {
        result.session_id = map.get("cookie");
    }

    std::string req_body = protocol::HttpUtil::decode_chunked_body(req);
    result.raw_body = req_body;

    auto parsed_req = jinq::common::parse_llm_chat_request(req_body);
    result.task_id = parsed_req.task_id;
    result.is_valid = parsed_req.is_valid;
    result.parse_status = parsed_req.parse_status;
    if (parsed_req.is_valid) {
        Dialog dialog;
        for (const auto& msg : parsed_req.messages) {
            dialog.push_back({msg.first, msg.second});
        }
        result.payload = dialog;
    }
    return result;
}

/***
 *
 * @param req
 * @param ctx
 */
void Llama3ChatServer::Impl::do_work(const task_request& req, std::shared_ptr<series_ctx> ctx) {
    // 解析失败：直接按解析错误码返回统一信封
    if (!req.is_valid) {
        {
            std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
            ctx->model_run_status = req.parse_status;
            ctx->task_finished_ts = Timestamp::now().to_format_str();
        }
        WFTaskFactory::count_by_name("release_ctx");
        return;
    }

    // 取单 worker：天然串行化聊天请求（会话亲和 + 互斥）
    LlamaModelPtr worker;
    _m_working_queue.wait_dequeue(worker);

    auto task_receive_ts = Timestamp::now();
    {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        // task 时间戳
        ctx->task_id = req.task_id;
        ctx->is_task_req_valid = req.is_valid;
        ctx->task_received_ts = task_receive_ts.to_format_str();
    }

    // 本轮新消息（KV cache 已保留历史上下文）
    auto current_dialog = std::any_cast<Dialog>(req.payload);

    // generate response
    auto status = worker->chat_completion(current_dialog, ctx->model_output);

    // 上下文超限时 shift kv cache 后重试
    if (status == StatusCode::LLM_CONTEXT_SIZE_EXCEEDED) {
        auto model_stat = worker->get_model_stat();
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
            status = worker->kv_cache_shift_top_n(n_discard, 0);
            if (status == StatusCode::OK) {
                break;
            }
        }
        if (try_times < 0) {
            LOG(ERROR) << "shift kv cache failed, clear kv cache";
            worker->clear_kv_cache_cell();
        }
        model_stat = worker->get_model_stat();
        LOG(INFO) << fmt::format("kv cache used cell counts: {} after shift", model_stat.kv_cache_cell_nums);
        LOG(INFO) << fmt::format("kv cache token counts: {} after shift", model_stat.kv_cache_token_nums);

        // re-generate response
        status = worker->chat_completion(current_dialog, ctx->model_output);
    }

    // cache history dialog
    if (status == StatusCode::OK) {
        Dialog turn_dialog = current_dialog;
        turn_dialog.push_back(ChatMessage({"assistant", ctx->model_output}));
        auto history_iter = _m_user_history_dialogs.find(req.session_id);
        if (history_iter != _m_user_history_dialogs.end()) {
            _m_user_history_dialogs[req.session_id] += turn_dialog;
        } else {
            _m_user_history_dialogs.insert(std::make_pair(req.session_id, turn_dialog));
        }
        // 回写会话 cookie，供客户端后续请求维持会话（与 do_work_cb 写 response 互斥）
        {
            std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
            ctx->response->add_header_pair("Set-Cookie", req.session_id);
        }
    }

    // restore worker queue
    update_status_snapshot(worker);
    _m_working_queue.enqueue(std::move(worker));

    {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->model_run_status = status;
        // update ctx
        auto task_finish_ts = Timestamp::now();
        ctx->task_finished_ts = task_finish_ts.to_format_str();
        ctx->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000;
    }
    WFTaskFactory::count_by_name("release_ctx");
}

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string Llama3ChatServer::Impl::make_response_body(
    const std::string& task_id,
    const StatusCode& status,
    const std::string& model_output) {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
    doc.AddMember("code", static_cast<int>(status), allocator);
    std::string msg = status == StatusCode::OK ? "success" : jinq::common::error_code_to_str(status);
    doc.AddMember("msg", rapidjson::Value(msg.c_str(), msg.size(), allocator), allocator);
    rapidjson::Value data;
    data.SetObject();
    data.AddMember("task_id", rapidjson::Value(task_id.c_str(), task_id.size(), allocator), allocator);
    data.AddMember("response", rapidjson::Value(model_output.c_str(), model_output.size(), allocator), allocator);
    doc.AddMember("data", data, allocator);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

/***
 *
 * @param task
 * @return
 */
bool Llama3ChatServer::Impl::handle_custom_endpoint(WFHttpTask* task) {
    const char* uri = task->get_req()->get_request_uri();

    if (strcmp(uri, "/check_model_stat") == 0) {
        // 只读快照：不触碰 worker 队列，handler 线程永不阻塞
        auto snapshot = read_status_snapshot();
        const auto& model_stat = snapshot.model_status;
        task->get_resp()->append_output_body(fmt::format(
            "<html>n_ctx: {}\n kv cache used: {}</html>",
            model_stat.n_ctx_size, model_stat.kv_cache_cell_nums));
        return true;
    } else if (strcmp(uri, "/clear_kv_cache") == 0) {
        // 清空 kv cache 需要独占 worker：异步 go task 排队执行，handler 线程立即返回
        auto* resp = task->get_resp();
        auto body = std::make_shared<std::string>();
        auto go_proc = [this, body]() {
            LlamaModelPtr worker;
            _m_working_queue.wait_dequeue(worker);
            worker->clear_kv_cache_cell();
            auto model_stat = worker->get_model_stat();
            update_status_snapshot(worker);
            _m_working_queue.enqueue(std::move(worker));
            *body = fmt::format(
                "<html>kv cache cleared.\n n_ctx: {}\n kv cache used: {}</html>",
                model_stat.n_ctx_size, model_stat.kv_cache_cell_nums);
        };
        auto* go_task = WFTaskFactory::create_go_task("clear_kv_cache", std::move(go_proc));
        go_task->set_callback([resp, body](const WFGoTask*) {
            resp->append_output_body(*body);
        });
        *series_of(task) << go_task;
        return true;
    } else if (strcmp(uri, "/get_context_perf") == 0) {
        // 只读快照
        auto snapshot = read_status_snapshot();
        const auto& data = snapshot.context_perf;
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
        return true;
    }
    return false;
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
StatusCode Llama3ChatServer::init(const toml::table &config) {
    // init impl
    auto status = _m_impl->init(config);
    if (status != StatusCode::OK) {
        LOG(INFO) << "init llama3 chat server failed";
        return status;
    }

    return init_http_server(_m_impl.get());
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
