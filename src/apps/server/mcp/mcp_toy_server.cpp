/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mcp_toy_server.cpp
 * Date: 25-4-8
 ************************************************/
// mcp toy server

#include <glog/logging.h>

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFHttpServer.h"
#include "workflow/WFFacilities.h"
#include "workflow/HttpUtil.h"

struct TaskInput {
    double a;
    double b;
};

TaskInput parse_task_input(const std::string& req_body) {
    rapidjson::Document doc;
    doc.Parse(req_body.c_str());
    TaskInput task_input {};
    CHECK_EQ(doc.IsObject(), true);
    task_input.a = doc["a"].GetDouble();
    task_input.b = doc["b"].GetDouble();

    return task_input;
}

std::string make_response(double result) {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();
    // write req id
    writer.Key("result");
    writer.Double(result);
    writer.EndObject();

    return buf.GetString();
}

void serve_process(WFHttpTask* task) {
    if (strcmp(task->get_req()->get_request_uri(), "/add") == 0) {
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto task_input = parse_task_input(protocol::HttpUtil::decode_chunked_body(req));
        auto result = task_input.a + task_input.b;
        auto resp_str = make_response(result);
        resp->append_output_body(resp_str);
    } else if (strcmp(task->get_req()->get_request_uri(), "/subtraction") == 0) {
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto task_input = parse_task_input(protocol::HttpUtil::decode_chunked_body(req));
        auto result = task_input.a - task_input.b;
        auto resp_str = make_response(result);
        resp->append_output_body(resp_str);
    } else if (strcmp(task->get_req()->get_request_uri(), "/multiplication") == 0) {
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto task_input = parse_task_input(protocol::HttpUtil::decode_chunked_body(req));
        auto result = task_input.a * task_input.b;
        auto resp_str = make_response(result);
        resp->append_output_body(resp_str);
    } else if (strcmp(task->get_req()->get_request_uri(), "/division") == 0) {
        auto* req = task->get_req();
        auto* resp = task->get_resp();
        auto task_input = parse_task_input(protocol::HttpUtil::decode_chunked_body(req));
        auto result = task_input.a / (task_input.b + 0.00000000000001);
        auto resp_str = make_response(result);
        resp->append_output_body(resp_str);
    } else {
        task->get_resp()->append_output_body("<html>404 Not Found</html>");
        return;
    }
}


int main(int argc, char** argv) {

    int port = 8090;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }

    WFFacilities::WaitGroup wait_group(1);
    auto server = std::make_unique<WFHttpServer>(serve_process);
    if (server->start(port) == 0) {
        wait_group.wait();
        server->stop();
    } else {
        LOG(ERROR) << "Cannot start server";
        return -1;
    }

    return 0;
}