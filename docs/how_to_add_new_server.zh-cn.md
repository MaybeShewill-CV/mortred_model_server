# 如何新增一个模型服务

本框架中新增一个 server 自注册表化重构后不再需要编写约 200 行的模板类：每个可服务模型是家族 `catalog()` 里的一条 `CvModelEntry`（两个 TOML 段名、worker 工厂、响应序列化器）。`ProductIndex` 把这一行投影到两个统一入口。通用实现位于 [jinq::server::CvModelServer&lt;MODEL_OUTPUT&gt;](../src/server/generic_cv_server.h)，它构建在 [jinq::server::BaseAiServerImpl&lt;WORKER, MODEL_OUTPUT&gt;](../src/server/base_server_impl.h) 之上，后者继续提供鉴权、限流、请求校验、单请求超时、worker 池、Prometheus 指标与 `/openapi.json` 端点。模型输入统一使用 base64 编码图像。下面以新增 densenet 图像分类服务为例；模型本身参考[如何新增模型](../docs/how_to_add_new_model.zh-cn.md)。

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

## 第 2 步：在 catalog 加一行

打开模型所属任务的工厂头文件（示例为 `src/factory/classification_task.h`），向 `catalog()` 追加一条 `CvModelEntry`。这一行就是产品身份：`model_section` 是 `--model` 键，`server_section` 对应 TOML 段，`make_worker` + `fill_response` 即可 HTTP 服务，家族默认 vis/默认图即可 bench。

```cpp
inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        // ...已有模型...
        Entry{"DENSENET", "densenet classification", "DENSENET_CLASSIFICATION_SERVER",
              &create_densenet_classifier<ImageInput, Output>,
              &jinq::server::response::fill_classification,
              classification_param_specs()},
    };
    return entries;
}
```

`ProductIndex` 启动时遍历各家族 `catalog()`。没有第二张产品表，也不再写 `create_densenet_cls_server` 薄包装。`CvModelServer<Output>::init` 会读取 server 段
（worker 池、超时、鉴权、限流，见[服务配置说明](../docs/about_model_server_configuration.zh-cn.md)）、
加载 model 段引用的模型配置、创建 `worker_nums` 个 worker 并装配 HTTP 服务。

如果输出是全新结构，先在 [src/server/response_serializers.h](../src/server/response_serializers.h)
补充对应序列化器并让 `fill_response` 指向它；字段名与 JSON 类型必须与
`docs/openapi.json` 的 `components.schemas` 一致（用 `python scripts/gen_openapi.py` 重新生成）。

## 第 3 步：统一入口，不再新建 ELF

不要再添加 `src/apps/server/*.cpp` 或 `add_mortred_app`。`ProductIndex` 会投影家族
catalog 里的每一行 `CvModelEntry`，因此 catalog 加一行之后即可：

```
mortred-model-server.out --model DENSENET /path/to/densenet_server_config.toml
mortred-model-benchmark.out --model DENSENET /path/to/densenet_config.toml [image]
```

然后：

1. 在 `conf/server/<task>/<model>/` 下添加配置，写上 `model = "DENSENET"` 和
   `server_exe = "mortred-model-server.out"`（复制同类配置并调整
   `server_uri`、`port`、`worker_nums`）；
2. `python3 scripts/check_consistency.py` 必须保持绿色（catalog id ↔ conf `model=`）；
3. 通过 `scripts/gen_openapi.py` 在 `docs/openapi.json` 中声明新的 `server_uri`。

## 第 4 步：框架已经替你做的事

通常不需要触碰请求服务逻辑。`BaseAiServerImpl` 提供
`serve_process` / `do_work` / `do_work_cb`：JSON 请求解析（含 400/413/415/405 契约错误）、
Bearer 鉴权、按 IP 限流、阻塞队列取 worker（计入超时预算）、模型推理、经
`fill_response` 的响应序列化、Prometheus 指标与结构化请求日志。仅当需要额外端点时
才使用扩展点（`handle_custom_endpoint`）。