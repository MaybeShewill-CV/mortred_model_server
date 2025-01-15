/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: qwen2_vl_chatbot_client.cpp
 * Date: 25-1-15
 ************************************************/
// qwen2-vl chat client tool

#include <regex>
#include <iostream>

#include <glog/logging.h>
#include "toml/toml.hpp"
#include "fmt/format.h"
#include "workflow/WFFacilities.h"
#include "workflow/HttpUtil.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/base64.h"
#include "common/file_path_util.h"

using jinq::common::Base64;
using jinq::common::FilePathUtil;

struct series_ctx {
    bool is_first_conversation=false;
    std::string url;
    std::string req_body;
    std::string uuid;
};

void request_chat_task(series_ctx* ctx);
void request_chat_task_cb(WFGoTask* task);
void print_server_response(WFHttpTask* task);

bool is_local_file(const std::string & url) {
    return FilePathUtil::is_file_exist(url);
};

bool is_url(const std::string & url) {
    std::vector<std::string> url_prefixes = {"http://", "https://", "ftp://"};
    return std::any_of(
        url_prefixes.begin(), url_prefixes.end(),
        [&](const std::string & protocol) -> bool {return url.find(protocol) != std::string::npos;});
};

bool is_b64(const std::string & url) {
    size_t len = url.length();
    if (len == 0 || len % 4 != 0) {
        return false;
    }

    int padding_count = 0;
    if (len >= 2 && url[len - 1] == '=') {
        padding_count++;
        if (url[len - 2] == '=') {
            padding_count++;
        }
    }

    auto is_base64_char = [](char c) -> bool {
        return (isalnum(c) || c == '+' || c == '/');
    };
    for (size_t i = 0; i < len - padding_count; ++i) {
        if (!is_base64_char(url[i])) {
            return false;
        }
    }

    for (size_t i = len - padding_count; i < len; ++i) {
        if (url[i] != '=') {
            return false;
        }
    }
    return true;
};

std::string parse_user_image(const std::string& image_url) {
    if (image_url.empty()) {
        return image_url;
    }

    if (is_b64(image_url) || is_url(image_url)) {
        return image_url;
    }

    if (is_local_file(image_url)) {
        std::ifstream f_in;
        f_in.open(image_url, std::ios::binary);
        if (!f_in.is_open()) {
            LOG(ERROR) << fmt::format("read image file: {} failed", image_url);
            return "";
        }
        std::ostringstream oss;
        oss << f_in.rdbuf();
        std::string encode_image_data = Base64::base64_encode(oss.str());
        return encode_image_data;
    } else {
        LOG(ERROR) << fmt::format(R"(not supported image url data, must be one of ['base64', 'web url', 'local file'])");
        return "";
    }
}

/***
 *
 * @param ctx
 */
void request_chat_task(series_ctx* ctx) {
    // prepare user input prompt
    LOG(INFO)<< "\033[32m>User Image:\n ----  \033[0m";
    std::string user_image;
    std::getline(std::cin, user_image);
    user_image = parse_user_image(user_image);

    LOG(INFO)<< "\033[32m>User Text:\n ----  \033[0m";
    std::string user_text;
    std::getline(std::cin, user_text);

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

        rapidjson::Value content_arr;
        content_arr.SetArray();
        rapidjson::Value text_cnt;
        text_cnt.SetObject();
        text_cnt.AddMember("type", "text", allocator);
        text_cnt.AddMember("text", rapidjson::Value(user_text.c_str(), allocator), allocator);
        content_arr.PushBack(text_cnt, allocator);
        rapidjson::Value image_cnt;
        image_cnt.SetObject();
        image_cnt.AddMember("type", "image", allocator);
        image_cnt.AddMember("image", rapidjson::Value(user_image.c_str(), allocator), allocator);
        content_arr.PushBack(image_cnt, allocator);

        user_data.AddMember("content", content_arr, allocator);
        data.PushBack(user_data, allocator);
        ctx->is_first_conversation=false;
    } else {
        rapidjson::Value user_data;
        user_data.SetObject();
        user_data.AddMember("role", rapidjson::Value("user", allocator), allocator);

        rapidjson::Value content_arr;
        content_arr.SetArray();
        rapidjson::Value text_cnt;
        text_cnt.SetObject();
        text_cnt.AddMember("type", "text", allocator);
        text_cnt.AddMember("text", rapidjson::Value(user_text.c_str(), allocator), allocator);
        content_arr.PushBack(text_cnt, allocator);
        rapidjson::Value image_cnt;
        image_cnt.SetObject();
        image_cnt.AddMember("type", "image", allocator);
        image_cnt.AddMember("image", rapidjson::Value(user_image.c_str(), allocator), allocator);
        content_arr.PushBack(image_cnt, allocator);

        user_data.AddMember("content", content_arr, allocator);
        data.PushBack(user_data, allocator);
    }
    doc.AddMember("data", data, allocator);
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
    if (!ctx->uuid.empty()) {
        req->add_header_pair("Cookie", ctx->uuid);
    }
    req->add_header_pair("Connection", "Keep-Alive");
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
    auto resp_content = doc["data"].GetObject()["response"].GetString();
    LOG(INFO)<< fmt::format("\033[33m>Server Response:\n ---- \n {} \n\033[0m", resp_content);

    // parse cookie
    protocol::HttpHeaderMap map(resp);
    if (map.key_exists("Set-Cookie")) {
        ctx->uuid = map.get("Set-Cookie");
    }

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

    if (argc != 2) {
        LOG(INFO) << "usage:";
        LOG(INFO) << "exe cfg_path";
        return -1;
    }

    static WFFacilities::WaitGroup wait_group(1);
    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config = toml::parse(config_file_path);
    const auto& server_cfg = config.at("QWEN2_VL_CHAT_SERVER");
    std::string host = server_cfg.at("host").as_string();
    int64_t port = server_cfg.at("port").as_integer();
    std::string uri = server_cfg.at("server_url").as_string();
    LOG(INFO) << "serve on host: " << host;
    LOG(INFO) << "serve on port: " << port;
    LOG(INFO) << "serve on uri: " << uri;

    auto* ctx = new series_ctx;
    ctx->is_first_conversation = true;
    ctx->url = fmt::format("http://{}:{}{}", host, port, uri);

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