/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: openai_compatible_chat_client.cpp
 * Date: 25-3-13
 ************************************************/
// simple openai compatible chat client

#include <iostream>

#include <glog/logging.h>
#include "toml/toml.hpp"
#include "fmt/format.h"
#include "workflow/WFGlobal.h"
#include "workflow/WFFacilities.h"
#include "workflow/HttpUtil.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"  // 美化输出

struct series_ctx {
    bool is_first_conversation=false;
    std::string url;
    std::string req_body;
    std::string model_name;
};

void request_chat_task(series_ctx* ctx);
void request_chat_task_cb(WFGoTask* task);
void print_server_response(WFHttpTask* task);

/***
 *
 * @param ctx
 */
void request_chat_task(series_ctx* ctx) {
    // prepare user input prompt
    LOG(INFO)<< "\033[32m>User Input:\n ----  \033[0m";
    std::string user;
    std::getline(std::cin, user);

    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
    rapidjson::Value data;
    data.SetArray();
    if (ctx->is_first_conversation) {
        rapidjson::Value sys_data;
        sys_data.SetObject();
        sys_data.AddMember("role", rapidjson::Value("system", allocator), allocator);
        sys_data.AddMember("content", rapidjson::Value("You are a smart ai assistant from Mortred Company. "
                                                       "Answer question shortly", allocator), allocator);
        data.PushBack(sys_data, allocator);
        rapidjson::Value user_data;
        user_data.SetObject();
        user_data.AddMember("role", rapidjson::Value("user", allocator), allocator);
        user_data.AddMember("content", rapidjson::Value(user.c_str(), allocator), allocator);
        data.PushBack(user_data, allocator);
        ctx->is_first_conversation=false;
    } else {
        rapidjson::Value user_data;
        user_data.SetObject();
        user_data.AddMember("role", rapidjson::Value("user", allocator), allocator);
        user_data.AddMember("content", rapidjson::Value(user.c_str(), allocator), allocator);
        data.PushBack(user_data, allocator);
    }
    doc.AddMember("messages", data, allocator);
    if (!ctx->model_name.empty()) {
        doc.AddMember("model", rapidjson::Value(ctx->model_name.c_str(), allocator), allocator);
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    auto request_body = buffer.GetString();
    ctx->req_body = request_body;
}

/***
 *
 * @param task
 */
void request_chat_task_cb(WFGoTask* task) {
    auto* ctx = (series_ctx*)series_of(task)->get_context();
    auto* http_task = WFTaskFactory::create_http_task(ctx->url, 1, 1, print_server_response);
    auto* req = http_task->get_req();
    req->set_method("POST");
    req->append_output_body(ctx->req_body);
    series_of(task)->push_back(http_task);
}

/***
 *
 * @param argc
 * @param argv
 * @return
 */
/***
 *
 * @param task
 */
void print_server_response(WFHttpTask* task) {
    auto state = task->get_state();
    auto error = task->get_error();
    auto* ctx = (series_ctx*)series_of(task)->get_context();

    // parse response
    if (state != WFT_STATE_SUCCESS) {
        auto err_msg = fmt::format("chat request task exec failed, state: {}, msg: {}",
                                   state, WFGlobal::get_error_string(state, error));
        LOG(ERROR) << "timeout reason: " << task->get_timeout_reason();
        LOG(ERROR) << err_msg;
        return;
    }
    auto* resp = task->get_resp();
    auto resp_body = protocol::HttpUtil::decode_chunked_body(resp);
    if (resp_body.empty()) {
        LOG(INFO)<< fmt::format("\033[33m>Server Response:\n ---- \n {} \n\033[0m", resp_body);
        auto* req_task = WFTaskFactory::create_go_task("req_task", request_chat_task, ctx);
        series_of(task)->push_back(req_task);
    }
    rapidjson::Document doc;
    doc.Parse(resp_body.c_str());
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    LOG(INFO)<< fmt::format("\033[33m>Server Response:\n ---- \n {} \n\033[0m", buffer.GetString());

    // fill in req task
    auto* req_task = WFTaskFactory::create_go_task("req_task", request_chat_task, ctx);
    req_task->set_callback(request_chat_task_cb);
    series_of(task)->push_back(req_task);
}

int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    if (argc != 2 && argc != 3) {
        LOG(INFO) << "usage:";
        LOG(INFO) << "exe host:port [model_name]";
        return -1;
    }

    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.endpoint_params.response_timeout = -1;
    WORKFLOW_library_init(&settings);

    static WFFacilities::WaitGroup wait_group(1);
    std::string host_port = argv[1];
    std::string uri = "/v1/chat/completions";
    LOG(INFO) << "serve on: " << host_port;
    std::string model_name = argc > 2 ? argv[2] : "";

    auto* ctx = new series_ctx;
    ctx->is_first_conversation = true;
    ctx->url = fmt::format("http://{}{}", host_port, uri);

    auto* series = Workflow::create_series_work(WFTaskFactory::create_empty_task(), [&](const SeriesWork* s_work) -> void {
        auto* tmp_ctx = (series_ctx*)s_work->get_context();
        delete tmp_ctx;
        wait_group.done();
    });
    series->set_context(ctx);
    auto* go_task = WFTaskFactory::create_go_task("req_task", request_chat_task, ctx);
    go_task->set_callback(request_chat_task_cb);
    series->push_back(go_task);
    series->start();
    wait_group.wait();

    return 0;
}