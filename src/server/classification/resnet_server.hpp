/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: resnet_server.hpp
 * Date: 22-6-19
 ************************************************/

#ifndef MM_AI_SERVER_RESNET_SERVER_H
#define MM_AI_SERVER_RESNET_SERVER_H

#include <atomic>
#include <sstream>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "toml/toml.hpp"
#include "workflow/HttpMessage.h"
#include "workflow/HttpUtil.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/Workflow.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "common/base64.h"
#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/md5.h"
#include "common/status_code.h"
#include "common/time_stamp.h"
#include "factory/classification_task.h"
#include "models/model_io_define.h"

namespace morted {
namespace server {

using morted::common::Base64;
using morted::common::CvUtils;
using morted::common::FilePathUtil;
using morted::common::Md5;
using morted::common::StatusCode;
using morted::common::Timestamp;

namespace classification {
using morted::factory::classification::create_resnet_classifier;
using morted::models::io_define::classification::std_classification_output;
using morted::models::io_define::common_io::base64_input;
using ResNet = morted::models::BaseAiModel<base64_input, std_classification_output>;
using ResNetPtr = std::unique_ptr<ResNet>;

struct TaskCount {
    std::atomic<size_t> recieved_jobs_ato{0};
    std::atomic<size_t> finished_jobs_ato{0};
    std::atomic<size_t> waiting_jobs_ato{0};
};

static TaskCount &get_task_count() {
    static TaskCount task_count;
    return task_count;
}

struct seriex_ctx {
    protocol::HttpResponse *response = nullptr;
};

struct ClsRequest {
    std::string image_content;
    std::string task_id;
    bool is_valid = true;
};

ClsRequest parse_task_request(const std::string &req_body) {
    rapidjson::Document doc;
    doc.Parse(req_body.c_str());
    ClsRequest req{};
    if (doc.HasParseError() || doc.IsNull() || doc.ObjectEmpty()) {
        req.image_content = "";
        req.is_valid = false;
    } else {
        CHECK_EQ(doc.IsObject(), true);

        if (!doc.HasMember("img_data") || !doc["img_data"].IsString()) {
            req.image_content = "";
            req.is_valid = false;
        } else {
            req.image_content = doc["img_data"].GetString();
            req.is_valid = true;
        }
        if (!doc.HasMember("req_id") || !doc["req_id"].IsString()) {
            req.task_id = "";
            req.is_valid = false;
        } else {
            req.task_id = doc["req_id"].GetString();
        }
    }
    return req;
}

std::string make_response_body(const std::string &task_id, const StatusCode &status, const std_classification_output &model_output) {

    int code = static_cast<int>(status);
    std::string msg = status == StatusCode::OK ? "success" : "fail";
    int cls_id = model_output.class_id;
    float scores = model_output.scores[cls_id];

    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    writer.StartObject();
    // write req id
    writer.Key("req_id");
    writer.String(task_id.c_str());
    // write code
    writer.Key("code");
    writer.Int(code);
    // write msg
    writer.Key("msg");
    writer.String(msg.c_str());
    // write class result
    writer.Key("data");
    writer.StartObject();
    writer.Key("class_id");
    writer.Int(cls_id);
    writer.Key("scores");
    writer.Double(scores);
    writer.EndObject();
    writer.EndObject();

    return buf.GetString();
}

static ResNetPtr &get_resnet_ptr(const std::string &model_name) {
    static ResNetPtr resnet_ptr = create_resnet_classifier<base64_input, std_classification_output>(model_name);
    if (resnet_ptr->is_successfully_initialized()) {
        return resnet_ptr;
    }
    std::string resnet_model_cfg_path = "../weights/classification/resnet50_config.ini";
    if (!FilePathUtil::is_file_exist(resnet_model_cfg_path)) {
        LOG(FATAL) << "resnet model config file not exist: " << resnet_model_cfg_path;
        resnet_ptr.reset(nullptr);
        return resnet_ptr;
    }
    auto cfg = toml::parse(resnet_model_cfg_path);
    resnet_ptr->init(cfg);
    if (!resnet_ptr->is_successfully_initialized()) {
        LOG(FATAL) << "resnet init failed";
        resnet_ptr.reset(nullptr);
    }
    return resnet_ptr;
}

void do_classification(const ClsRequest &req, seriex_ctx *ctx) {
    // get task receive timestamp
    auto task_receive_ts = Timestamp::now();
    // get resnet model
    auto &classifier = get_resnet_ptr("resnet50");
    // get task count
    auto &task_count = get_task_count();
    // construct model input
    base64_input model_input{req.image_content};
    // do classification
    std_classification_output model_output;
    auto status = classifier->run(model_input, model_output);
    if (status != StatusCode::OK) {
        LOG(ERROR) << "classifier run failed";
    }
    // make response body
    std::string response_body = make_response_body(req.task_id, status, model_output);
    // fill response
    ctx->response->append_output_body(response_body);
    // update task count
    task_count.finished_jobs_ato++;
    task_count.waiting_jobs_ato--;
    // output log info
    auto task_finish_ts = Timestamp::now();
    auto task_elapse_ts = task_finish_ts - task_receive_ts;
    LOG(INFO) << "task id: " << req.task_id << " received at: " << task_receive_ts.to_format_str()
              << " finished at: " << task_finish_ts.to_format_str() << " elapse: " << task_elapse_ts << " ms";
}

void server_process(WFHttpTask *task) {
    // welcome message
    if (strcmp(task->get_req()->get_request_uri(), "/welcome") == 0) {
        DLOG(INFO) << "Request-URI: " << task->get_req()->get_request_uri();
        task->get_resp()->append_output_body("<html>Welcome to Morted Resnet classification Server</html>");
        return;
    }
    // hello world message
    if (strcmp(task->get_req()->get_request_uri(), "/hello_world") == 0) {
        DLOG(INFO) << "Request-URI: " << task->get_req()->get_request_uri();
        task->get_resp()->append_output_body("<html>Hello World !!!</html>");
        return;
    }
    // resnet classification
    if (strcmp(task->get_req()->get_request_uri(), "/morted_ai_server_v1/classification/resnet") == 0) {
        DLOG(INFO) << "Request-URI: " << task->get_req()->get_request_uri();
        // parse request body
        auto *req = task->get_req();
        auto *resp = task->get_resp();
        auto cls_task_req = parse_task_request(protocol::HttpUtil::decode_chunked_body(req));
        // update task count
        auto &task_count = get_task_count();
        task_count.waiting_jobs_ato++;
        task_count.recieved_jobs_ato++;
        // init series work
        auto *series = series_of(task);
        auto *ctx = new seriex_ctx;
        ctx->response = resp;
        series->set_context(ctx);
        series->set_callback([](const SeriesWork *series) { delete (seriex_ctx *)series->get_context(); });
        // do classification
        auto *cls_task = WFTaskFactory::create_go_task("resnet_cls_work", do_classification, cls_task_req, ctx);
        *series << cls_task;
    }
}

} // namespace classification
} // namespace server
} // namespace morted

#endif // MM_AI_SERVER_RESNET_SERVER_H