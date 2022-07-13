# How To Add New Server

Here is brief instruction about how to add a new server in this frame work. All model server are inherited from [jinq::server::BaseAiServer](../src/server/abstract_server.h) which determin the server's interface function. You're supposed to pay more attention to [jinq::server::BaseAiServerImpl<WORKER, MODEL_OUTPUT>](../src/server/base_server_impl.h). It's the actual implemention of the base server and all specific servers are inherited from this implementation. `WORKER` is the model in this framework for example the new densenet image classification model in [how_to_add_new_model.md](../docs/how_to_add_new_model.md). `MODEL_OUTPUT` is model's output defined by users. The input of the model uses base64 encoded images uniformly considring convenience and efficiency. Server's main process consist three major module first parse client's request data and fetch base64 encoded input image second send that image into worker queue waiting to run inference finally make response and reply to the client. I will show you an example to help you add a new densenet image classification server in the next sections.

## Step 1: Define Your Own Output Data Type :monkey_face:

This step is the same as adding a new model. Default model's output data type for different kind of vision tasks can be found in [model_io_define.h](../src/models/model_io_define.h). Those structs which are named after std** represent the default model output. 

For example the default model output for classification task is

```cpp
namespace classification {
    struct cls_output {
        int class_id;
        std::vector<float> scores;
    };
    using std_classification_output = cls_output;
} 
```

class_id equals max score's idx in scores.

## Step 2: Inherit Your New Model Server

The new model server inherit from [jinq::server::BaseAiServer](../src/server/abstract_server.h) and is only response for interface. The class private member `_m_impl` is responsible for actual implementation. Detailed code can be found [densenet_server.h](../src/server/classification/densenet_server.h). Main structure is

```cpp
namespace jinq {
namespace server {
namespace classification {
class DenseNetServer : public jinq::server::BaseAiServer {
public:
    DenseNetServer();

    ~DenseNetServer() override;

    DenseNetServer(const DenseNetServer& transformer) = delete;

    DenseNetServer& operator=(const DenseNetServer& transformer) = delete;

    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg) override;

    void serve_process(WFHttpTask* task) override;

    bool is_successfully_initialized() const override;

private:
    class Impl;
    std::unique_ptr<Impl> _m_impl;
};
}
}
}
```

Private class member `_m_impl` inherit from [BaseAiServerImpl<WORKER, MODEL_OUTPUT>](../src/server/base_server_impl.h). Here `WORKER` represent the densenet model which can be create by factory function and `MODEL_OUTPUT` use the default classification model's output so `DenseNetServer::Impl` can be constructed like

```cpp
using jinq::models::io_define::classification::std_classification_output;
using jinq::factory::classification::create_densenet_classifier;
using DenseNetPtr = decltype(create_densenet_classifier<base64_input, std_classification_output>(""));

class DenseNetServer::Impl : public BaseAiServerImpl<DenseNetPtr, std_classification_output>
```

Detailed code can be found [densenet_server.cpp#L40-L61](../src/server/classification/densenet_server.cpp)

## Step 3: Implement SubClass Specific Interface

Each server impl subclass should implement two specific interface

```cpp
/***
*
* @param config
* @return
*/
StatusCode init(const decltype(toml::parse(""))& config) override;

/***
 *
 * @param task_id
 * @param status
 * @param model_output
 * @return
 */
std::string make_response_body(const std::string& task_id, const StatusCode& status, const std_classification_output& model_output) override;
```

`init` interface is used to initialize model server due to each server's specific configuration. You may checkout [about_model_server_configuration.md](../docs/about_model_server_configuration.md) for server's configuration details.

`make_response_body` is used to transfor model's output into response content. User must implement the interface function consdering each server may have special response format.

## Step 4: Implment SubClass Server Interface Function

That interface can be directly inherit from base class's implementation. Major module of server process is first `parse client request` second run model session which is packaged into a `WFGoTask` finally make response and reply to client which is implemented in `WFGoTask_CallBack Function`. Main server's code structure is

```cpp
/***
 * parse client request and start a go task to run model session
 * @tparam WORKER
 * @tparam MODEL_INPUT
 * @tparam MODEL_OUTPUT
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::serve_process(WFHttpTask* task) {
    // parse client request
    auto* req = task->get_req();
    auto* resp = task->get_resp();
    auto cls_task_req = parse_task_request(protocol::HttpUtil::decode_chunked_body(req));

    // construct a go task to run model session
    auto* series = series_of(task);

    auto&& go_proc = std::bind(&BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work, this, std::placeholders::_1, std::placeholders::_2);
    WFGoTask* serve_task = WFTaskFactory::create_go_task(_m_server_uri, go_proc, cls_task_req, ctx);
    auto&& go_proc_cb = std::bind(&BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work_cb, this, serve_task);
    serve_task->set_callback(go_proc_cb);
    *series << serve_task;

    return;
}

/***
 * run model session and get model output
 * @tparam WORKER
 * @tparam MODEL_INPUT
 * @tparam MODEL_OUTPUT
 * @param req
 * @param ctx
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work(const BaseAiServerImpl::cls_request& req, BaseAiServerImpl::seriex_ctx* ctx) {
    // fetch a model worker from worker_queue
    WORKER worker;
    while (!_m_working_queue.try_dequeue(worker)) {}

    // run model session
    models::io_define::common_io::base64_input model_input{req.image_content};
    StatusCode status = worker->run(model_input, ctx->model_output);

    // return work back to queue
    while (!_m_working_queue.enqueue(std::move(worker))) {}

    ...
}

/***
 * make response and reply to client
 * @tparam WORKER
 * @tparam MODEL_INPUT
 * @tparam MODEL_OUTPUT
 * @param task
 */
template<typename WORKER, typename MODEL_OUTPUT>
void BaseAiServerImpl<WORKER, MODEL_OUTPUT>::do_work_cb(const WFGoTask* task) {
    ...
    // make response body
    auto* ctx = (seriex_ctx*)series_of(task)->get_context();
    StatusCode status = ctx->model_run_status;
    std::string task_id = ctx->is_task_req_valid ? ctx->task_id : "";
    std::string response_body = make_response_body(task_id, status, ctx->model_output);

    // reply to client
    ctx->response->append_output_body(std::move(response_body));

    ...
}
```

You'd better use the base class's implementation if you're a beginner otherwise you can implement your own serve logic if you've got some specific needs.

## Step 5: Add New Server Into Task Factory :factory:

Task factory is used to create model object. Details about this can be found [classification_task.inl#L100-L108](../src/factory/classification_task.h)

```cpp
/***
 * create densenet image classification server
 * @param detector_name
 * @return
 */
static std::unique_ptr<BaseAiServer> create_densenet_cls_server(const std::string& server_name) {
    REGISTER_AI_SERVER(DenseNetServer, server_name)
    return ServerFactory<BaseAiServer>::get_instance().get_server(server_name);
}
```

## Step 6: Make A New Server App :airplane:

You have already add a new ai model server till this step. Now let's make a server app to verify. Comple code can be found [densenet_classification_server.cpp](../src/apps/server/classification/densenet_classification_server.cpp)

```cpp
int main(int argc, char** argv) {

    static WFFacilities::WaitGroup wait_group(1);

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config = toml::parse(config_file_path);
    const auto &server_cfg = config.at("DENSENET_CLASSIFICATION_SERVER");
    auto port = server_cfg.at("port").as_integer();
    LOG(INFO) << "serve on port: " << port;

    auto server = create_densenet_cls_server("densenet_cls_server");
    server->init(config);
    if (server->start(port) == 0) {
        wait_group.wait();
        server->stop();
    } else {
        LOG(ERROR) << "Cannot start server";
        return -1;
    }

    return 0;
}
```

If nothing wrong happened :smile: you should get a similar server described in [toturials_of_classification_model_server](../docs/toturials_of_classification_model_server.md)

Good Luck !!! :trophy::trophy::trophy:

## Reference

Complete implementation code can be found

* [Base Server Impl Implement](../src/server/base_server_impl.h)
* [DenseNet Server Implement](../src/server/classification/densenet_server.cpp)
* [DenseNet Server App](../src/apps/server/classification/densenet_classification_server.cpp)
