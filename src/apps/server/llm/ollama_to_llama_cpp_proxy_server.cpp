/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: ollama_to_llama_cpp_proxy_server.cpp
 * Date: 25-4-28
 ************************************************/

// proxy server

#include <csignal>
#include <charconv>
#include <cstdlib>
#include <cstdio>
#include <string_view>
#include <utility>
#include "workflow/Workflow.h"
#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFHttpServer.h"
#include "workflow/WFFacilities.h"
#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/request_size_limit.h"

struct tutorial_series_context {
    std::string url;
    WFHttpTask *proxy_task = nullptr;
    bool is_keep_alive = true;
};

void reply_callback(WFHttpTask *proxy_task) {
    SeriesWork *series = series_of(proxy_task);
    auto *context =
        (tutorial_series_context *)series->get_context();
    auto *proxy_resp = proxy_task->get_resp();
    size_t size = proxy_resp->get_output_body_size();

    if (proxy_task->get_state() == WFT_STATE_SUCCESS) {
        fprintf(stderr, "%s: Success. Http Status: %s, BodyLength: %zu\n", context->url.c_str(), proxy_resp->get_status_code(), size);
    } else /* WFT_STATE_SYS_ERROR*/ {
        fprintf(stderr, "%s: Reply failed: %s, BodyLength: %zu\n", context->url.c_str(), strerror(proxy_task->get_error()), size);
    }
}

void http_callback(WFHttpTask *task) {
    int state = task->get_state();
    int error = task->get_error();
    auto *resp = task->get_resp();
    SeriesWork *series = series_of(task);
    auto *context = (tutorial_series_context *)series->get_context();
    auto *proxy_resp = context->proxy_task->get_resp();
    auto llm_decode_str = protocol::HttpUtil::decode_chunked_body(task->get_resp());

    LOG(INFO) << "qwen resp: " << llm_decode_str;

    if (state == WFT_STATE_SUCCESS) {
        const void *body;
        size_t len;

        /* set a callback for getting reply status. */
        context->proxy_task->set_callback(reply_callback);

        /* Copy the remote webserver's response, to proxy response. */
        resp->get_parsed_body(&body, &len);
        resp->append_output_body_nocopy(body, len);
//        *proxy_resp = std::move(*resp);

        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

        doc.AddMember("model", "qwen3-14b", allocator);
        doc.AddMember("created_at", "2023-08-04T08:52:19.385406455-07:00", allocator);
        rapidjson::Value message;
        message.SetObject();
        message.AddMember("role", "assistant", allocator);
        message.AddMember("content", rapidjson::Value(llm_decode_str.c_str(), allocator), allocator);
        doc.AddMember("message", message, allocator);
        doc.AddMember("done", true, allocator);
        doc.AddMember("total_duration", 0, allocator);
        doc.AddMember("load_duration", 0, allocator);
        doc.AddMember("prompt_eval_count", 0, allocator);
        doc.AddMember("prompt_eval_duration", 0, allocator);
        doc.AddMember("eval_count", 0, allocator);
        doc.AddMember("eval_duration", 0, allocator);
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        proxy_resp->append_output_body(buffer.GetString());
        if (!context->is_keep_alive) {
            proxy_resp->set_header_pair("Connection", "close");
        }
    } else {
        const char *err_string;

        if (state == WFT_STATE_SYS_ERROR) {
            err_string = strerror(error);
        } else if (state == WFT_STATE_DNS_ERROR) {
            err_string = gai_strerror(error);
        } else if (state == WFT_STATE_SSL_ERROR) {
            err_string = "SSL error";
        } else /* if (state == WFT_STATE_TASK_ERROR) */ {
            err_string = "URL error (Cannot be a HTTPS proxy)";
        }

        fprintf(stderr, "%s: Fetch failed. state = %d, error = %d: %s\n",
                context->url.c_str(), state, error, err_string);
        LOG(ERROR) << err_string;

        /* As a tutorial, make it simple. And ignore reply status. */
        proxy_resp->set_status_code("404");
        proxy_resp->append_output_body_nocopy("<html>404 Not Found.</html>", 27);
    }
}

void process(WFHttpTask *proxy_task) {
    auto *req = proxy_task->get_req();
    SeriesWork *series = series_of(proxy_task);
    WFHttpTask *http_task; /* for requesting remote webserver. */

    auto *context = new tutorial_series_context;

    std::string ori_uri = req->get_request_uri();

    LOG(INFO) << "request uri: " << ori_uri;
//    LOG(INFO) << "request params: " << protocol::HttpUtil::decode_chunked_body(req);

    std::string proxy_base_uri = "http://127.0.0.1:8080";
    std::string proxy_url;
    if (ori_uri == "/api/chat") {
        proxy_url = proxy_base_uri + "/v1/chat/completions";
    } else if (ori_uri == "/api/tags") {
        std::string tag_resp = "{\n"
                               "  \"models\": [\n"
                               "    {\n"
                               "      \"name\": \"codellama:13b\",\n"
                               "      \"modified_at\": \"2023-11-04T14:56:49.277302595-07:00\",\n"
                               "      \"size\": 7365960935,\n"
                               "      \"digest\": \"9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697\",\n"
                               "      \"details\": {\n"
                               "        \"format\": \"gguf\",\n"
                               "        \"family\": \"llama\",\n"
                               "        \"families\": null,\n"
                               "        \"parameter_size\": \"13B\",\n"
                               "        \"quantization_level\": \"Q4_0\"\n"
                               "      }\n"
                               "    },\n"
                               "    {\n"
                               "      \"name\": \"llama3:latest\",\n"
                               "      \"modified_at\": \"2023-12-07T09:32:18.757212583-08:00\",\n"
                               "      \"size\": 3825819519,\n"
                               "      \"digest\": \"fe938a131f40e6f6d40083c9f0f430a515233eb2edaa6d72eb85c50d64f2300e\",\n"
                               "      \"details\": {\n"
                               "        \"format\": \"gguf\",\n"
                               "        \"family\": \"llama\",\n"
                               "        \"families\": null,\n"
                               "        \"parameter_size\": \"7B\",\n"
                               "        \"quantization_level\": \"Q4_0\"\n"
                               "      }\n"
                               "    }\n"
                               "  ]\n"
                               "}";
        proxy_task->get_resp()->append_output_body_nocopy(tag_resp.c_str());
        return;
    }

    context->url = proxy_url;
    context->proxy_task = proxy_task;

    series->set_context(context);
    series->set_callback([](const SeriesWork *series) {
        delete (tutorial_series_context *)series->get_context();
    });

    context->is_keep_alive = req->is_keep_alive();
//    http_task = WFTaskFactory::create_http_task(req->get_request_uri(), 0, 0, http_callback);
    http_task = WFTaskFactory::create_http_task(proxy_url, 0, 0, http_callback);

    const void *body;
    size_t len;

    /* Copy user's request to the new task's reuqest using std::move() */
    req->set_request_uri(http_task->get_req()->get_request_uri());
    req->get_parsed_body(&body, &len);
    req->append_output_body_nocopy(body, len);
    *http_task->get_req() = std::move(*req);

    /* also, limit the remote webserver response size. */
    http_task->get_resp()->set_size_limit(200 * 1024 * 1024);

    *series << http_task;
}

static WFFacilities::WaitGroup wait_group(1);

void sig_handler(int signo)
{
    wait_group.done();
}

int main(int argc, char *argv[])
{
    unsigned short port;

    if (argc != 2)
    {
        fprintf(stderr, "USAGE: %s <port>\n", argv[0]);
        exit(1);
    }

    std::string_view port_arg(argv[1]);
    auto [ptr, ec] = std::from_chars(port_arg.data(), port_arg.data() + port_arg.size(), port);
    if (ec != std::errc() || ptr != port_arg.data() + port_arg.size() || port == 0) {
        fprintf(stderr, "USAGE: %s <port>\n", argv[0]);
        exit(1);
    }
    signal(SIGINT, sig_handler);

    struct WFServerParams params = HTTP_SERVER_PARAMS_DEFAULT;
    /* for safety, limit request size to the default 64MB. */
    params.request_size_limit = jinq::common::k_default_request_size_limit_mb * 1024 * 1024;

    WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
    settings.endpoint_params.response_timeout = -1;
    WORKFLOW_library_init(&settings);

    WFHttpServer server(&params, process);

    if (server.start("127.0.0.1", port) == 0) {
        wait_group.wait();
        server.stop();
    } else {
        perror("Cannot start server");
        exit(1);
    }

    return 0;
}
