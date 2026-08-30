# P4：Modern Model Developer Experience 改造计划

> 状态：实施中
> 执行清单：[model-developer-experience-todolist.zh-cn.md](model-developer-experience-todolist.zh-cn.md)
> 基线：`main @ 5648960 refactor(models): validate latent and clip outputs`  
> 建议分支：`refactor/models-p4-developer-experience`

## 1. 背景与目标

P0 到 P2 已完成模型层正确性基础：

- 每次推理拥有独立的 `InferenceContext`；
- 请求几何从 preprocessing 贯通到 postprocessing；
- MNN / ONNX Runtime / TensorRT 由统一 backend session 接入；
- f32 输出具备 dtype、rank、shape、buffer 和 finite value 契约；
- object detection、classification、segmentation、OCR、matting、enhancement、feature point、depth、FastSAM、CLIP 和 diffusion 已接入相应契约。

当前短板不再是正确性，而是开发体验：

- 新增模型仍需要修改过多位置；
- 模型作者仍频繁手写图像转换、tensor 打包和 `memcpy`；
- factory/server 注册代码重复度高；
- golden 与 contract 测试仍以手工复制为主；
- 模型 IO 定义集中在一个大文件中；
- 参数读取、错误提示和范围校验风格不统一。

P4 的目标是：

> 让新增普通 CV 模型时，开发者只关注模型本身的 preprocess、参数和 decode；生命周期、契约、注册、测试骨架和文档骨架由基础设施提供。

## 2. 设计原则

### 2.1 保持简单

P4 不引入重型框架，不做动态插件系统，不追求“一切皆配置”。优先提供：

- 小而清晰的 runtime toolkit；
- 显式模型注册 catalog；
- 可执行的脚手架；
- 可复用的测试工具；
- 少量高收益的模型族迁移。

### 2.2 保持既有正确性契约

所有新 API 必须构建在现有基础设施上：

- `InferenceContext`
- `TensorContract`
- `f32_output.h`
- `request_geometry.h`
- `BackendCvModel`

P4 不能降低 P0/P2 已达到的安全边界。

### 2.3 不隐藏复杂 decode

模型差异大的 decode 逻辑保留在各模型中。公共层只承接确定性重复逻辑：

- 图像转换与 tensor 打包；
- 参数读取；
- 输出读取与契约校验；
- request geometry；
- 常见分类、mask、alpha 输出转换。

不把 YOLO、SAM、diffusion 等差异强行塞入通用基类。

### 2.4 不立即升级 C++20

当前项目基线为 C++17，兼容性优先。P4 不依赖：

- C++20 concepts；
- `std::span`；
- C++ modules；
- 动态反射。

可以在 C++17 内通过小型 value type、builder 和静态断言达到目标。

## 3. 当前新增模型路径

当前新增一个模型通常需要接触以下位置：

| 序号 | 位置 | 内容 |
|---|---|---|
| 1 | `src/models/<task>/*.h` | 模型类与成员声明 |
| 2 | `src/models/<task>/*.inl` | preprocess、postprocess、on_init |
| 3 | `conf/model/<task>/<model>/*.toml` | backend 与模型参数 |
| 4 | `src/factory/<task>_task.h` | model factory 与 server factory |
| 5 | server spec | section、display name、serializer、worker maker |
| 6 | `test/*_unittest.cc` | contract 或行为测试 |
| 7 | `test/model_golden_test.cc` | golden case |
| 8 | `test/golden/*` | golden 数据 |
| 9 | weights / TRT profile / manifest | 权重与 engine 配置 |
| 10 | 文档 | 模型说明与接入说明 |

其中真正模型相关的只有：

- preprocess；
- decode/postprocess；
- 模型参数；
- 输出契约；
- golden 行为。

其余流程应模板化或注册化。

## 4. P4 目标开发路径

P4 完成后，普通单图像模型的理想接入路径为：

```text
1. 运行 scripts/new_model.py 生成模型骨架
2. 实现 preprocess pipeline
3. 实现 output reader + decode
4. 补模型参数与输出契约
5. 添加 catalog entry
6. 补 contract test 与 golden case
```

目标文件：

```text
src/models/<task>/<model_family>/<model>.h
src/models/<task>/<model_family>/<model>.inl
conf/model/<task>/<model_family>/<model>.toml
test/<model>_output_contract_unittest.cc
test/golden/<model>.json
src/models/catalog/<task>.cpp
```

普通模型不再要求修改：

- backend session 实现；
- HTTP server 实现；
- factory 中重复的 server 模板代码；
- tensor 内存拷贝细节；
- 多个无关任务的 IO 定义。

## 5. 分阶段改造计划

### Phase 0：基线与度量

**任务**

1. 固化当前模型层指标：
   - 模型数量；
   - 新增模型平均修改文件数；
   - `std::memcpy` 出现次数；
   - 手写 `convert_to_chw_vec` 次数；
   - 手写 output contract 次数；
   - factory/server 重复代码规模；
   - tests-only 与 full build 编译时间。
2. 写一个开发体验基准文档。
3. 明确每阶段必须保持绿色：
   - tests-only；
   - `-Werror`；
   - CPU profile；
   - full GPU golden；
   - sanitizers。

**交付**

```text
docs/model-developer-experience-metrics.zh-cn.md
```

**验收**

- 指标可由脚本重复采集；
- 不改变运行行为；
- main CI 全绿。

### Phase 1：Model Runtime Toolkit

新增：

```text
src/models/backend/model_runtime.h
src/models/backend/model_runtime.cpp
test/model_runtime_unittest.cc
```

#### 5.1 ImagePipeline

目标 API：

```cpp
auto input = ImagePipeline(image)
                 .bgr_to_rgb()
                 .resize(network_size)
                 .to_float()
                 .scale(1.0f / 255.0f)
                 .mean_std(mean, std)
                 .nchw(input_name);
```

支持：

- 空图检查；
- color conversion；
- resize；
- uint8 到 float；
- scale；
- mean/std；
- NCHW / NHWC 打包；
- 连续 buffer 拷贝；
- shape 与 byte size 校验；
- 统一错误信息。

返回使用 C++17 可实现的 `RuntimeResult<T>`，不引入异常，不改变现有 `StatusCode` 边界。

#### 5.2 OutputReader

目标 API：

```cpp
auto scores = OutputReader(outputs, "output")
                  .f32()
                  .shape({1, class_count})
                  .finite();

if (!scores.ok()) {
    return scores.status();
}
```

底层复用：

- `validated_f32_named_output`
- `TensorContract`
- `require_finite_f32`

必须保留语义：

```text
missing output -> MODEL_EMPTY_OUTPUT
dtype/rank/shape/buffer 错误 -> MODEL_OUTPUT_CONTRACT_FAILED
NaN/Inf -> MODEL_OUTPUT_CONTRACT_FAILED
```

#### 5.3 ParamReader

目标 API：

```cpp
auto result = ParamReader(params, "MY_MODEL")
                  .get("score_threshold", &score_threshold)
                  .min(0.0)
                  .max(1.0);
if (!result.ok()) {
    return result.status();
}
```

统一能力：

- required / optional；
- default；
- int / int64 / float / double / bool / string；
- range；
- non-empty string；
- 数组长度；
- 未知 key 检查可配置；
- 统一错误前缀。

#### 5.4 SessionIoValidator

目标 API：

```cpp
auto input_info = SessionIoValidator(session())
                      .input("images")
                      .dtype(DType::F32)
                      .rank(4)
                      .nchw()
                      .channels(3)
                      .validate();
```

用于替代各模型 `on_init` 中重复的 session input shape 检查。

**Phase 1 验收**

- runtime toolkit 单测完整；
- 不迁移模型也可合入；
- 不改变现有 golden；
- 新 helper 无裸指针逃逸；
- `-Werror` 通过。

### Phase 2：模型目录与注册治理

新增：

```text
src/models/catalog/model_entry.h
src/models/catalog/classification.cpp
src/models/catalog/object_detection.cpp
src/models/catalog/scene_segmentation.cpp
src/models/catalog/ocr.cpp
src/models/catalog/matting.cpp
src/models/catalog/enhancement.cpp
src/models/catalog/feature_point.cpp
src/models/catalog/depth.cpp
src/models/catalog/sam.cpp
src/models/catalog/clip.cpp
src/models/catalog/diffusion.cpp
```

`ModelEntry` 包含：

```cpp
struct ModelEntry {
    std::string model_section;
    std::string server_section;
    std::string display_name;
    std::string task;
    ModelCreator creator;
    ResponseSerializerKind serializer;
};
```

每个任务维护一个显式 catalog，不使用全局静态注册，避免初始化顺序和隐藏副作用。

Factory 从 catalog 读取 entry，统一生成：

- model creator；
- server creator；
- `CvServerSpec`；
- worker maker；
- serializer 绑定。

原手写函数保留兼容包装，逐步废弃。

新增：

```text
test/model_catalog_unittest.cc
```

验证：

- model section 非空且唯一；
- server section 非空且唯一；
- display name 非空；
- 对应模型配置存在；
- 对应 server 配置存在；
- serializer 合法；
- creator 可创建对象；
- catalog 覆盖现有 factory 暴露模型。

**Phase 2 落地结果（as-built）**

实际实现比原方案更克制，避免为单一形态造抽象：

```text
src/models/catalog/model_entry.h     # ModelEntry / ServedModelEntry + 校验函数
src/factory/cv_catalog.h             # CvModelEntry<OUTPUT> + create_server
src/factory/model_catalog.h          # ModelCatalogEntry<INPUT, OUTPUT> + create_model
src/factory/<task>_task.h            # 每个任务自己的 catalog()，全部 header-only
```

- catalog 是任务内的 `inline catalog()` 函数，不拆成独立 `.cpp`，也没有全局静态注册。
- `ModelEntry` 只保留 `model_section + display_name`；`ServedModelEntry` 再加
  `server_section`；`CvModelEntry<OUTPUT>` 再加 worker creator 与 response filler。
  没有引入 `task` 字符串、`ModelCreator` 类型擦除或 `ResponseSerializerKind` 枚举。
- 没有 HTTP 面的模型族（CLIP、SAM predictor、FastSAM）用 `ModelCatalogEntry`，
  不强行补 server section。
- object detection / face detection 输出契约不同，拆成 `catalog()` 与 `face_catalog()`。
- diffusion 4 个 sampler 共用 base64 adapter，统一挂到通用 CV server。
- `test/model_catalog_unittest.cc` 额外校验 server TOML 中的
  `model_config_file_path` 指向的文件真实存在，并实际构造 `CvModelServer<OUTPUT>`。

**Phase 2 验收**

- 新增普通模型只需一个 catalog entry；
- factory 重复代码显著减少；
- 现有模型创建行为不变；
- model catalog 测试通过；
- e2e 与 golden 不漂移。

### Phase 3：脚手架生成器

新增：

```text
scripts/new_model.py
templates/model/*.h.in
templates/model/*.inl.in
templates/model/*.toml.in
templates/model/*_contract_unittest.cc.in
templates/model/README.md.in
```

使用方式：

```bash
python scripts/new_model.py \
  --task object_detection \
  --name rtdetr \
  --family rtdetr \
  --backend tensorrt \
  --input image \
  --output boxes
```

生成：

1. 模型类骨架；
2. preprocess TODO；
3. output contract TODO；
4. decode TODO；
5. TOML 配置；
6. catalog entry 提示；
7. contract test；
8. golden 文件占位提示；
9. 模型文档骨架；
10. 验证命令。

**Phase 3 验收**

- 生成后的 tests-only 编译可通过；
- TODO 标记清晰；
- 不覆盖已有文件，除非显式 `--force`；
- 支持 `--dry-run`；
- 支持 `--list-tasks`；
- 脚本自身有自测；
- `scripts/check_consistency.py` 能识别生成物。

### Phase 4：模型 IO 拆分

将 `model_io_define.h` 拆为：

```text
src/models/io/common_input.hpp
src/models/io/classification.hpp
src/models/io/object_detection.hpp
src/models/io/face_detection.hpp
src/models/io/scene_segmentation.hpp
src/models/io/ocr.hpp
src/models/io/matting.hpp
src/models/io/enhancement.hpp
src/models/io/feature_point.hpp
src/models/io/depth.hpp
src/models/io/clip.hpp
src/models/io/sam.hpp
src/models/io/diffusion.hpp
```

保留兼容聚合头 `model_io_define.h`。

**Phase 4 验收**

- 旧 include 路径不变；
- 新任务只修改自己的 IO 文件；
- 编译依赖减少；
- tests-only 编译时间可测量下降；
- 现有测试全部通过。

### Phase 5：测试基础设施注册化

新增：

```text
test/model_contract_test_util.h
test/model_golden_registry.h
```

Contract 测试目标：

```cpp
POSTPROCESS_CONTRACT_TEST(
    MyModel,
    "output",
    Shape{1, 1000},
    DType::F32,
    Context{.source = {640, 480}, .network = {224, 224}});
```

自动覆盖：

- missing output；
- wrong dtype；
- wrong rank；
- wrong shape；
- short buffer；
- NaN；
- Inf；
- contract status；
- output 不被部分污染。

Golden 注册目标：

```cpp
GOLDEN_CLASSIFICATION_CASE(
    "mobilenetv2_classification",
    "conf/model/classification/mobilenetv2/mobilenetv2_config.toml",
    "demo_data/model_test_input/classification/xxx.JPEG",
    create_mobilenetv2_classifier,
    "mobilenetv2_classification");
```

统一处理：

- weights 缺失时 skip；
- config path 修正；
- backend 强制 CPU；
- golden 文件路径；
- 容差选择；
- 结果比较；
- 失败输出。

**Phase 5 验收**

- 现有 golden 用例名称不变；
- golden 输出不漂移；
- 新增 golden case 不再复制流程代码；
- contract 异常矩阵自动生成；
- 测试文件行数下降。

### Phase 6：按模型族渐进迁移

迁移顺序：

1. classification；
2. matting；
3. scene segmentation；
4. enhancement；
5. OCR；
6. object detection；
7. feature point；
8. depth；
9. FastSAM；
10. CLIP；
11. diffusion。

选择该顺序的原因：

- 先迁移结构简单的单 session 模型；
- 再迁移输出形态相近的 dense output 模型；
- 最后处理多 session、多阶段和 latent 模型。

每个模型族迁移：

```text
旧 preprocess -> ImagePipeline
手写 memcpy -> InputTensor builder
手写 TOML parse -> ParamReader
手写 output contract -> OutputReader
手写 session shape check -> SessionIoValidator
```

每个模型族验收：

```text
model family contract tests 通过
CPU profile full check 通过
full GPU -Werror 编译通过
相关 GPU golden 不漂移
sanitizers 通过
```

### Phase 7：多 session 模型模板

适用：

- SAM encoder + decoder；
- SAM AMG 多 decoder；
- LightGlue extractor + matcher；
- CLIP visual encoder + text encoder；
- diffusion sampler + VAE decoder。

新增：

```text
src/models/backend/session_group.h
```

目标 API：

```cpp
SessionGroup sessions;
sessions.declare("encoder", "encoder_backend");
sessions.declare("decoder", "decoder_backend");
```

统一处理：

- session 声明；
- config 读取；
- session 创建；
- init 失败统一 reset；
- session 生命周期；
- 错误日志。

**Phase 7 验收**

- 多 session 模型 init 代码减少；
- 失败路径不泄漏 session；
- SAM / CLIP golden 不漂移；
- 不引入新的基类继承层级。

### Phase 8：文档与开发者引导

更新：

```text
docs/how_to_add_new_model.md
docs/how_to_add_new_model.zh-cn.md
docs/model-contract-governance.md
README.md
README.zh-cn.md
```

新增开发者路径：

- 10 分钟接入最小分类模型；
- 10 分钟接入 detection 模型；
- 如何写 output contract；
- 如何补 golden；
- 何时不应使用公共 helper；
- 如何调试 shape/dtype 错误。

**Phase 8 验收**

- 新开发者按文档可以在不阅读 backend 实现的情况下完成模型骨架；
- 文档中的命令可执行；
- 文档明确标注当前阶段与未完成项；
- P4 文档与实际 API 保持一致。

## 6. 代码设计草案

### 6.1 RuntimeResult

```cpp
template <typename T>
struct RuntimeResult {
    StatusCode status = StatusCode::OK;
    std::string error;
    T value{};

    bool ok() const { return status == StatusCode::OK; }
};
```

要求：

- 不使用异常；
- 不替代对外 `StatusCode` API；
- 错误信息一次性传递；
- 移动友好。

### 6.2 ImagePipeline

```cpp
class ImagePipeline {
  public:
    explicit ImagePipeline(const cv::Mat &image);

    ImagePipeline &bgr_to_rgb();
    ImagePipeline &rgb_to_bgr();
    ImagePipeline &resize(const cv::Size &size);
    ImagePipeline &to_float();
    ImagePipeline &scale(float factor);
    ImagePipeline &mean_std(const std::array<float, 3> &, const std::array<float, 3> &);

    RuntimeResult<NamedTensor> nchw(const std::string &name) const;
    RuntimeResult<NamedTensor> nhwc(const std::string &name) const;
};
```

约束：

- pipeline 内部状态只表示中间图像；
- 不保存请求级 source size；
- 生成 tensor 后立即返回；
- 不修改原始输入 Mat。

### 6.3 OutputReader

```cpp
class OutputReader {
  public:
    OutputReader(const std::vector<NamedTensor> &outputs, const std::string &name);

    OutputReader &f32();
    OutputReader &shape(std::vector<int64_t> shape);
    OutputReader &finite();

    RuntimeResult<F32OutputView> read() const;
};
```

注意：

- `F32OutputView` 生命周期仍由调用方 outputs vector 持有；
- reader 不复制大 buffer；
- reader 不放宽既有 contract。

### 6.4 ParamReader

```cpp
class ParamReader {
  public:
    ParamReader(const toml::table &params, std::string log_prefix);

    ParamReader &get(const std::string &key, int *value);
    ParamReader &get(const std::string &key, int64_t *value);
    ParamReader &get(const std::string &key, float *value);
    ParamReader &get(const std::string &key, double *value);
    ParamReader &get(const std::string &key, bool *value);
    ParamReader &get(const std::string &key, std::string *value);

    ParamReader &min(double value);
    ParamReader &max(double value);
    ParamReader &non_empty();

    RuntimeResult<void> validate() const;
};
```

### 6.5 ModelEntry

由于现有 `BaseAiModel` 是模板类型，catalog 初期按任务拆分，避免一开始就设计全任务类型擦除。

## 7. 明确不做的事情

P4 明确不做：

1. 不引入 Spring 式 DI 容器；
2. 不做运行时 C++ 插件加载；
3. 不做完全 YAML/DSL 化模型定义；
4. 不把所有 decode 抽到基类；
5. 不引入跨任务全局万能 Model 基类；
6. 不机械重命名全部 `_m_` 成员；
7. 不为追求语法现代而升级 C++20；
8. 不在 P4 中混入性能优化或部署改造；
9. 不用大规模 `std::any` 掩盖类型边界；
10. 不牺牲现有 golden 稳定性换取代码量下降。

## 8. 测试与验收矩阵

每个阶段至少执行：

```text
python scripts/check_consistency.py
python scripts/gen_openapi.py --check
clang-format --dry-run --Werror <changed files>
```

涉及代码时执行：

```text
tests-only check
tests-only -Werror check
CPU profile full check
full GPU -Werror build
相关 contract tests
完整 GPU golden
TSAN
ASan/UBSan
```

最终验收：

```text
CI 全绿
CodeQL 全绿
CPU profile 全绿
full Werror 全绿
完整 GPU golden 全绿
无 golden drift
模型 catalog 覆盖率 100%
新增普通模型接触点 <= 5
模型源码中手写 memcpy 显著减少
脚手架生成后可直接编译
新模型文档路径完整
```

## 9. 量化目标

| 指标 | 当前目标 |
|---|---|
| 新增普通模型修改文件数 | 从约 8-10 个降到 3-5 个 |
| 模型手写 `std::memcpy` | 仅保留特殊模型，普通模型为 0 |
| 手写 CHW/HWC 打包 | 普通图像模型为 0 |
| factory/server 重复代码 | 减少 60% 以上 |
| contract test 样板 | 由模板自动生成异常矩阵 |
| golden case 样板 | 每个 case 降低到一次注册 |
| 新模型初始编译失败成本 | 脚手架生成后即可编译 |
| 新开发者需要理解的 backend 细节 | 接近 0 |

## 10. 风险与控制

| 风险 | 控制措施 |
|---|---|
| 过度抽象 | 每阶段只解决已量化问题，不做万能框架 |
| 行为漂移 | 每个模型族迁移后跑 golden |
| API 复杂化 | 新 API 必须比旧代码更短、更明确 |
| 编译时间上升 | toolkit 放 `.cpp`，避免 header-only 大实现 |
| catalog 初始化顺序问题 | 显式 catalog，不做全局静态注册 |
| 类型擦除过度 | catalog 按任务拆分，保留模板边界 |
| 测试迁移风险 | golden case 名称和文件保持不变 |
| 文档漂移 | 文档示例必须来自可编译代码或测试 |

## 11. 建议实施顺序

```text
1. Phase 0: metrics
2. Phase 1: runtime toolkit
3. Phase 2: catalog
4. Phase 3: scaffolder
5. Phase 4: IO split
6. Phase 5: test registry
7. Phase 6: migrate simple families
8. Phase 7: multi-session template
9. Phase 8: docs and adoption guide
```

建议拆分 PR：

```text
PR 1: P4 metrics + runtime toolkit
PR 2: model catalog
PR 3: scaffolder
PR 4: IO split
PR 5: golden/contract test registry
PR 6-10: model family migration
PR 11: multi-session template
PR 12: developer docs
```

## 12. 与生产化阶段的关系

P4 聚焦模型开发体验。生产化阶段聚焦：

- 性能基准；
- worker/batch 策略；
- 资源画像；
- 部署升级；
- 监控告警；
- release 与 rollback。

推荐执行顺序：

```text
P4 先落地基础 toolkit 和 catalog
后续生产化阶段再基于稳定 catalog 做全量模型 benchmark
```

这样 benchmark 可以自动遍历 catalog，不需要再维护一份模型清单。
