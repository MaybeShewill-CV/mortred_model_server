# 如何添加新的DL模型

> 本页是 [how_to_add_new_model.md](how_to_add_new_model.md) 的中文版，
> 讲解统一后端层的结构与生命周期。按任务一步步操作的中文指南见
> [model-developer-guide.md](model-developer-guide.md)（英文，路径导向）。
>
> 当前实现基于 `BackendCvModel`。旧版文档描述的
> "自定义输入 → 模型内部输入 → session → 模型内部输出 → 自定义输出"
> 五层转换已经不存在，本页已按现行 API 重写。

## Step 0：用脚手架生成样板（推荐）

```bash
# 查看支持的任务
python scripts/new_model.py --list-tasks

# 预览将要生成的文件，不落盘
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet \
    --backend mnn --dry-run

# 实际生成
python scripts/new_model.py --task classification \
    --name efficientnet --class EfficientNet \
    --backend mnn
```

会生成 5 个文件：

| 文件 | 需要你填的内容 |
|---|---|
| `src/models/classification/<file>.h` | 类骨架，已继承 `BackendCvModel` |
| `src/models/classification/<file>.inl` | `preprocess` / `postprocess` / `on_init` 三个钩子 |
| `conf/model/classification/<name>/<name>_config.toml` | `[SECTION]` + `.backend` + `.params` |
| `test/<file>_output_contract_unittest.cc` | 输出契约测试 |
| `docs/models/classification/<name>.md` | 文档骨架 |

此时模型**可以直接编译**，所有钩子返回 `MODEL_NOT_IMPLEMENTED`，
半成品不会被误当成能跑的模型启动。
`src/models/object_detection/rtdetr_detector.*` 是一个已入库的生成样例，
同时充当模板可编译性的哨兵。

脚手架还会打印两段它**刻意不自动应用**的片段：catalog 条目和测试目标注册。

## Step 1：选择 IO 类型

IO 类型在 [src/models/io/](../src/models/io) 下，每个任务一个头文件。
`common_input.h` 存放共享输入（`mat_input` / `file_input` / `base64_input` /
`pair_mat_input`），各任务头文件存放自己的 `std_*_output`。

只 include 你需要的那个任务头。旧的
[model_io_define.h](../src/models/model_io_define.h) 仍然可用，
但它是一个会把所有任务都拉进来的兼容聚合头。

可加载的图像输入走默认的 `prepare_inputs` 路径；
任务默认输出（`std_*_output`）是推荐选择。

## Step 2：实现模型类

先读这几个参考实现：

- [mobilenetv2](../src/models/classification/mobilenetv2.inl) —— MNN 单图分类
- [yolov8_detector](../src/models/object_detection/yolov8_detector.inl) —— TensorRT 检测（decode + NMS）
- [ddpm_unet](../src/models/diffusion/ddpm_unet.inl) —— ONNX Runtime 非图像输入（重写 `prepare_inputs`）

```cpp
template <typename INPUT, typename OUTPUT>
class MyModel : public jinq::models::BackendCvModel<INPUT, OUTPUT> {
  public:
    MyModel() : jinq::models::BackendCvModel<INPUT, OUTPUT>("MY_MODEL") {}

  private:
    // 图像 -> 命名输入张量（图像模型必须实现）
    std::vector<jinq::models::backend::NamedTensor> preprocess(const cv::Mat &image) override;

    // 命名输出张量 + 请求几何 -> 任务输出
    jinq::common::StatusCode postprocess(const std::vector<jinq::models::backend::NamedTensor> &outputs,
                                         const jinq::models::backend::InferenceContext &context,
                                         OUTPUT &output) override;

    // 可选：读取 [MY_MODEL.params] 中的模型专属键
    jinq::common::StatusCode on_init(const toml::table &params) override;
};
```

要点：

- 构造函数传入配置 section 名（`"MY_MODEL"`）。
- **优先用 [runtime toolkit](../src/models/backend/model_runtime.h)**，不要手写这些步骤：
  - `ImagePipeline`：resize / 裁剪 / 颜色转换 / 归一化，一次调用打包成 NCHW 或 NHWC
  - `OutputReader`：替代手搭的输出契约
  - `SessionIoValidator`：替代手写的 session shape 检查
  - `ParamReader`：带范围诊断的 TOML 参数读取

  [src/models/](../src/models) 下每个已迁移的模型都是例子。
- `session().inputs()` / `session().outputs()` 暴露
  `TensorInfo{name, dtype, shape, dynamic}`；从它推导网络输入尺寸，不要写死。
- 用 `backend::Tensor::make<float>(shape)` 产生张量；shape 必须具体，
  并与模型文件期望的布局一致（MNN TENSORFLOW 导出配 `input_layout = "nhwc"`，
  TensorRT / ONNX 是 nchw）。
- 多输出模型按 `name` 取张量（见
  [tensor_contract.h](../src/models/backend/tensor_contract.h) 的 `find_output`）。
- 目标检测模型复用
  [detector_common.h](../src/models/object_detection/detector_common.h)：
  请求几何缩放、命名 f32 输出校验、NMS / top-k / 类别填充、NCHW 打包。
  模型专属的 decode 逻辑留在检测器里，不要挪到又一个基类后面。
- 非图像输入（token id、latent 向量、图像对）重写 `prepare_inputs`，
  不重写 `preprocess`。
- 请求作用域的数据（源图尺寸、网络尺寸、裁剪几何）必须放在
  `InferenceContext` 里；**绝不**存成模型成员，
  因为一个 worker 可能正在处理动态 batch。
- 稠密图像输出（mask、alpha、深度、增强图）只有在源几何校验通过之后
  才 resize 到 `context.source_size`。
  产生坐标的模型用共享的 request-geometry helper，不要除以输入尺寸成员。
- 通过 [f32_output.h](../src/models/backend/f32_output.h) 校验后端输出后再 decode。
  畸形张量必须返回 `MODEL_OUTPUT_CONTRACT_FAILED`，
  而不是产出一个解了一半的任务结果。
- 多引擎模型（encoder + decoder）配置 `<key>_backend` 子表，
  用 `make_session("<key>_backend")` 创建额外 session，
  并在 `run_sessions` 重写里编排它们。
- 如果引擎是**固定数量、各自独立、按名字索引**的一组 session，
  改继承 [MultiSessionModel](../src/models/backend/multi_session_model.h)，
  用 `sessions()` 声明，而不是手写 create / validate / reset 序列。
  它**刻意不做**运行编排。

## Step 3：写配置

```toml
[MY_MODEL]
[MY_MODEL.backend]
type = "mnn"                # mnn | onnx | tensorrt
model_file_path = "../weights/my_model/model.mnn"
device = "gpu"             # cpu | gpu
threads = 4
gpu_mem_limit_mb = 2048     # 仅 onnx+cuda；0 = 不限制；默认 2048
input_layout = "nhwc"       # 仅 mnn: auto | nhwc | nchw

[MY_MODEL.params]
score_threshold = 0.25
```

完整键参考见 [about_model_configuration.md](about_model_configuration.md)。
旧的 `BACKEND_DICT` / `XXX_TRT` / `XXX_ONNX` / `XXX_MNN` 三段式配置已经移除；
用 [scripts/migrate_model_config.py](../scripts/migrate_model_config.py) 迁移
（先 `--dry-run`，CI 里用 `--check`）。

## Step 4：在任务 catalog 里注册

每个任务在 `src/factory/<task>_task.h` 里有一个显式 catalog。
新增一个被服务的模型现在是一行加一个 creator——
没有手写的 server 注册 lambda，也没有复制的 `CvServerSpec` 块：

```cpp
// src/factory/my_task.h
template <typename INPUT, typename OUTPUT>
std::unique_ptr<BaseAiModel<INPUT, OUTPUT>> create_my_model(const std::string &name) {
    (void)name;
    return std::make_unique<MyModel<INPUT, OUTPUT>>();
}

using Output = jinq::models::io_define::my_task::std_my_task_output;
using Entry = jinq::factory::cv_catalog::CvModelEntry<Output>;

inline const std::vector<Entry> &catalog() {
    static const std::vector<Entry> entries = {
        Entry{"MY_MODEL", "My model display name", "MY_MODEL_SERVER",
              &create_my_model<jinq::server::Base64Input, Output>,
              &jinq::server::response::fill_my_task},
    };
    return entries;
}
```

剩下的交给 `factory::cv_catalog::create_server(catalog(), "MY_MODEL", server_name)`：
它在 `ServerFactory<BaseAiServer>` 里注册 creator，
并构建通用的 `CvModelServer<Output>`。

刻意存在两种形态：

- [factory/cv_catalog.h](../src/factory/cv_catalog.h) ——
  挂在通用 CV server 上的模型（`CvModelEntry<OUTPUT>` 携带 worker creator 和
  response filler）
- [factory/model_catalog.h](../src/factory/model_catalog.h) ——
  只被 benchmark 和进程内调用方直接消费、还没有 HTTP 面的模型族
  （CLIP、SAM predictor、FastSAM）

一个任务有多个输出契约时，**按契约拆成多个 typed catalog**，
不要合并成一个类型擦除的列表——
见 [obj_detection_task.h](../src/factory/obj_detection_task.h) 里的
`catalog()` 与 `face_catalog()`。

`test/model_catalog_unittest.cc` 会在 catalog 行引用了不存在的 TOML section
或 `model_config_file_path`，或 model / server section 跨任务重复时报错。

## Step 5：验证

```bash
cmake --preset full && cmake --build --preset full
scripts/run_tests.sh build/full -R model_golden_test --output-on-failure
```

用 [model_golden_registry.h](../test/model_golden_registry.h) 里的一个宏
注册 golden 用例（完整列表和两命令基线流程见
[开发者指南](model-developer-guide.md)）。
用 [golden_drift_check.py](../scripts/golden_drift_check.py)
证明重构没有改变任何数值，
用 [POSTPROCESS_CONTRACT_TEST](../test/model_contract_test_util.h)
覆盖七项拒绝矩阵。

容差按任务定：检测用 score / box-IoU，稠密输出用指纹 diff。

## 参考

- [模型开发者指南](model-developer-guide.md) —— 六条任务路径、helper 边界、调试
- [模型契约治理](model-contract-governance.md) —— 评审清单
- [P4 改造计划](model-developer-experience-p4.zh-cn.md) —— 本次重构的设计与各阶段记录
