# 如何新增一个模型服务

本框架中新增一个 server 自注册表化重构后不再需要编写约 200 行的模板类：每个服务只是一条 `AiServerSpec` 注册项（两个 TOML 段名、worker 工厂、响应序列化器），通过 creator 闭包注册。通用实现位于 [jinq::server::AiModelServer&lt;MODEL_OUTPUT&gt;](../src/server/generic_ai_server.h)，它构建在 [jinq::server::BaseAiServerImpl&lt;WORKER, MODEL_OUTPUT&gt;](../src/server/base_server_impl.h) 之上，后者继续提供鉴权、限流、请求校验、单请求超时、worker 池、Prometheus 指标与 `/openapi.json` 端点。模型输入统一使用 base64 编码图像。下面以新增 densenet 图像分类服务为例；模型本身参考[如何新增模型](../docs/how_to_add_new_model.zh-cn.md)。

## 第 1 步：定义输出数据类型 :monkey_face:

与新增模型一致。各视觉任务的默认输出类型定义在 [model_io_define.h](../src/models/model_io_define.h) 中，以 `std_*_output` 命名。分类任务默认输出：

```cpp
namespace classification {
    struct cls_output {
        int class_id;
        std::vector<float> scores;
    };
    using std_classification_output = cls_output;
}
```

`class_id` 等于 `scores` 中最大分数的下标。如果你的任务需要新的输出结构，先在此定义——server spec 与响应序列化器都引用它。

## 第 2 步：注册一条 AiServerSpec

打开模型所属任务的工厂头文件（示例为 `src/factory/classification_task.h`），将 server 创建函数写为 spec 闭包注册：

```cpp
// create densenet classification server
inline std::unique_ptr<BaseAiServer> create_densenet_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        using Output = jinq::models::io_define::classification::std_classification_output;
        jinq::server::AiServerSpec<Output> spec;
        spec.server_section = "DENSENET_CLASSIFICATION_SERVER";  // server TOML 段
        spec.model_section = "DENSENET";                          // 含 model_config_file_path
        spec.display_name = "Densenet classification";
        spec.make_worker = [](const std::string& name) {
            return create_densenet_classifier<jinq::server::Base64Input, Output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_classification;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<Output>(std::move(spec)));
    });
    return server_factory.create(server_name);
}
```

这就是新增一个服务的全部代码量。`AiModelServer<Output>::init` 会读取 server 段
（worker 池、超时、鉴权、限流，见[服务配置说明](../docs/about_model_server_configuration.zh-cn.md)）、
加载 model 段引用的模型配置、创建 `worker_nums` 个 worker 并装配 HTTP 服务——
即此前 22 个手写类重复的那套流程。

如果输出是全新结构，先在 [src/server/response_serializers.h](../src/server/response_serializers.h)
补充对应序列化器并让 `spec.fill_response` 指向它；字段名与 JSON 类型必须与
`docs/openapi.json` 的 `components.schemas` 一致（用 `python scripts/gen_openapi.py` 重新生成）。

## 第 3 步：接线可执行入口

在 `src/apps/server/<task>/` 下新建薄壳 main（路径与可执行名属于仓库布局契约，
`scripts/check_consistency.py` 会校验），委托给公共入口：

```cpp
// densenet classification server tool

#include "apps/common/model_server_main.h"
#include "factory/classification_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "DENSENET_CLASSIFICATION_SERVER",
        [](const std::string& server_name) {
            return jinq::factory::classification::create_densenet_cls_server(server_name);
        });
}
```

然后：

1. 在 `src/apps/CMakeLists.txt` 中添加目标（`add_server_app(...)`）；
2. 在 `conf/server/<task>/<model>/` 下添加配置（复制同类配置并调整
   `server_uri`、`port`、`worker_nums`）；
3. 在 `docs/repository-layout.md` 中补充可执行文件行（一致性门禁会校验该映射，
   以及 `model_config_file_path` 指向的文件存在）；
4. 通过 `scripts/gen_openapi.py` 在 `docs/openapi.json` 中声明新的 `server_uri`。

## 第 4 步：框架已经替你做的事

通常不需要触碰请求服务逻辑。`BaseAiServerImpl` 提供
`serve_process` / `do_work` / `do_work_cb`：JSON 请求解析（含 400/413/415/405 契约错误）、
Bearer 鉴权、按 IP 限流、阻塞队列取 worker（计入超时预算）、模型推理、经
`fill_response` 的响应序列化、Prometheus 指标与结构化请求日志。仅当需要额外端点时
才使用扩展点（`handle_custom_endpoint`）。