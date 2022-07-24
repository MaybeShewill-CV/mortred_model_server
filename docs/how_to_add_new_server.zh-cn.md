# 如何增加新的DL模型服务

下面是一个简要教程教大家如何快速新增一个DL模型服务. 所有的模型服务类都继承自 [jinq::server::BaseAiServer](../src/server/abstract_server.h) 接口类。除去接口用户更应该关心的是这个实现类 [jinq::server::BaseAiServerImpl<WORKER, MODEL_OUTPUT>](../src/server/base_server_impl.h)，它负责了整个服务类的功能实现. `WORKER` 指代的是该框架内的DL模型，例如在文档 [how_to_add_new_model.md](../docs/how_to_add_new_model.md) 中新增的 `DenseNet` 图像分类模型. `MODEL_OUTPUT` 是指代用户自定义的模型输出. 出于效率和便捷性考虑，DL模型输入统一使用 `base64` 编码的图像。服务的主要处理过程可以抽象为三步，首先解析客户端发送的请求并获取其中的base64编码图像内容，其次将任务丢入 `worker_queue` 中等待模型计算处理，最后获取模型输出并将其填入response返回给客户端。接下来将通过一个例子讲述如何将之前新增的 `densenet` 分类模型包装成为服务

## Step 1: 定义模型的输出数据格式 :monkey_face:

This step is the same as adding a new model. Default model's output data type for different kind of vision tasks can be found in [model_io_define.h](../src/models/model_io_define.h). Those structs which are named after std** represent the default model output.

这一步和新增DL模型中的步骤是完全相同的. 例如默认的分类模型输出为

```cpp
namespace classification {
    struct cls_output {
        int class_id;
        std::vector<float> scores;
    };
    using std_classification_output = cls_output;
} 
```

class_id 等于scores序列中最大值的索引号.

## Step 2: 继承一个新的Server类

新的Server类继承自 [jinq::server::BaseAiServer](../src/server/abstract_server.h) 并且仅仅用来定义统一接口。该类的private成员 `_m_impl` 用来负责具体的服务实现. 代码细节可参考 [densenet_server.h](../src/server/classification/densenet_server.h). 主要的结构如下

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

私有成员 `_m_impl` 继承自 [BaseAiServerImpl<WORKER, MODEL_OUTPUT>](../src/server/base_server_impl.h)。`WORKER` 模板参数代表着任务工厂所能创建的DL模型，`MODEL_OUTPUT` 代表用户自定义的模型输出。整体 `DenseNetServer::Impl` 实现主体结构如下

```cpp
using jinq::models::io_define::classification::std_classification_output;
using jinq::factory::classification::create_densenet_classifier;
using DenseNetPtr = decltype(create_densenet_classifier<base64_input, std_classification_output>(""));

class DenseNetServer::Impl : public BaseAiServerImpl<DenseNetPtr, std_classification_output>
```

代码细节可参考 [densenet_server.cpp#L40-L61](../src/server/classification/densenet_server.cpp)

## Step 3: 实现子类的接口函数

每个Server实现子类都需要实现这两个特殊的接口函数。

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

`init` 负责根据每个Server不同的配置参数来初始化Server. 关于Server参数配置可以查看文档 [about_model_server_configuration.md](../docs/about_model_server_configuration.zh-cn.md)。

`make_response_body` 接口负责将模型的输出转换成服务端的response信息返回给客户端。

## Step 4: 实现子类的 `serve_process` 接口函数 （可选）

该函数主要负责服务端的serve逻辑，如果没有特殊需求一般可以直接继承自基类. 该函数的主要过程分为三步，首选通过 `parse client request` 来获取客户端发送的图像数据，其次是调用模型inference过程，在这里该计算任务被包装成一个 `WFGoTask`，最后在 `WFGoTask_CallBack Function` 计算任务的回调函数中将模型的输出转换成服务端的response信息返回给客户端。主要的代码结构如下

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

如果是新手的话推荐直接继承基类的实现过程，如果你有特殊的需求那么可以自行修改实现过程。

## Step 5: 任务工厂中增加创建接口 :factory:

在任务工厂中创建相应的接口函数，细节实现可以参考 [classification_task.inl#L100-L108](../src/factory/classification_task.h)

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

## Step 6: 创建一个新的图像分类服务app :airplane:

至此为止你已经成功添加了一个全新的图像分类服务。现在就来创建一个分类服务app来测试下效果吧. 完整的代码可参考 [densenet_classification_server.cpp](../src/apps/server/classification/densenet_classification_server.cpp)

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

不出意外 :smile: 你就可以得到一个和文档 [toturials_of_classification_model_server](../docs/toturials_of_classification_model_server.md) 中描述相同的高性能图像分类服务器了。

祝你好运 !!! :trophy::trophy::trophy:

## 参考

完整代码如下

* [Base Server Impl Implement](../src/server/base_server_impl.h)
* [DenseNet Server Implement](../src/server/classification/densenet_server.cpp)
* [DenseNet Server App](../src/apps/server/classification/densenet_classification_server.cpp)
