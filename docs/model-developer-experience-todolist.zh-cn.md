# P4 Modern Model Developer Experience TODO List

> 执行分支：`refactor/models-p4-developer-experience`  
> 方案文档：[model-developer-experience-p4.zh-cn.md](model-developer-experience-p4.zh-cn.md)  
> 勾选规则：只有该步骤代码、测试、文档和必要验证全部完成后才标记为完成。

## 总览

| 阶段 | 状态 |
|---|---|
| Phase 0 基线与度量 | 已完成 |
| Phase 1 Runtime Toolkit | 已完成 |
| Phase 2 Model Catalog | 已完成 |
| Phase 3 Scaffolder | 已完成 |
| Phase 4 IO 拆分 | 已完成 |
| Phase 5 测试注册化 | 已完成 |
| Phase 6 模型族迁移 | 未开始 |
| Phase 7 多 Session 模板 | 未开始 |
| Phase 8 文档与引导 | 未开始 |

## 准备工作

- [x] 固化 P4 总体方案文档
- [x] 创建专用开发分支 `refactor/models-p4-developer-experience`
- [x] Phase 0 完成前确认 main 基线 CI 状态
  - 备注：`main @ 7a68279` 的 tests / quality / sanitizers / CPU profile 已通过；full Werror Docker 与 CodeQL full 的阻塞已定位为 GCC 9 下 `DepthAnything::postprocess` 的 unused-variable，并已在本分支移除。

## Phase 0：基线与度量

- [x] 新增 `scripts/model_dx_metrics.py`
  - [ ] 统计模型源码文件与行数
  - [ ] 统计模型实现中的 `std::memcpy`
  - [ ] 统计手写 `convert_to_chw_vec`
  - [ ] 统计手写 `Tensor::make<float>`
  - [ ] 统计 f32 output contract 调用
  - [ ] 统计 factory/server 重复注册结构
  - [ ] 支持 JSON / Markdown 输出
- [x] 新增 `docs/model-developer-experience-metrics.zh-cn.md`
- [x] 记录 Phase 0 基线指标
- [x] 本地运行脚本并验证输出稳定

## Phase 1：Model Runtime Toolkit

- [x] 新增 `src/models/backend/model_runtime.h`
- [x] 新增 `src/models/backend/model_runtime.cpp`
- [x] 实现 `RuntimeResult<T>`
- [x] 实现 `ImagePipeline`
  - [x] 空图检查
  - [x] color conversion
  - [x] resize
  - [x] center crop
  - [x] uint8 -> float
  - [x] scale
  - [x] mean/std
  - [x] NCHW 打包
  - [x] NHWC 打包
- [x] 实现 `OutputReader`
  - [x] named output
  - [x] f32 dtype contract
  - [x] rank / shape contract
  - [x] finite 校验
- [x] 实现 `ParamReader`
  - [x] int / int64 / float / double / bool / string
  - [x] min / max
  - [x] non-empty
  - [x] required 基础实现（optional 语法保持待后续补充）
- [x] 实现 `SessionIoValidator`
  - [x] input / output name
  - [x] dtype
  - [x] rank
  - [x] NCHW / NHWC
  - [x] channels
  - [x] static/dynamic shape
- [x] 新增 `test/model_runtime_unittest.cc`
- [x] 注册 `model_runtime_unittest`
- [x] 用 MobileNetv2 classification 试点替换手写 preprocess / output contract

## Phase 2：Model Catalog

- [x] 设计任务内 `ModelEntry`
- [x] 新增 `src/models/catalog/model_entry.h`
- [x] 建立 classification catalog
- [x] 建立 object detection catalog
- [x] 建立 scene segmentation catalog
- [x] 建立 OCR catalog
- [x] 建立 matting catalog
- [x] 建立 enhancement catalog
- [x] 建立 feature point catalog
- [x] 建立 depth catalog
- [x] 建立 SAM catalog
- [x] 建立 CLIP catalog
- [x] 建立 diffusion catalog
- [x] factory 从 catalog 创建模型
- [x] server spec 从 catalog 驱动
- [x] 新增 `test/model_catalog_unittest.cc`
- [x] 校验 section 唯一性
- [x] 校验配置文件存在
- [x] 校证 catalog 覆盖现有 factory 暴露模型


> Phase 2 落地说明：
> - 所有任务目录都是显式的任务内 `catalog()` 函数，没有全局静态注册副作用。
> - `models::catalog::ModelEntry` 是最小模型身份（model section + display name），
>   `ServedModelEntry` 额外携带 server section；`factory::cv_catalog::CvModelEntry<OUTPUT>`
>   再补上 worker creator 与 response filler。
> - 没有通用 CV server 的模型族（CLIP、SAM predictor、FastSAM）使用
>   `factory::model_catalog::ModelCatalogEntry`，不强行造出 server section。
> - object detection 与 face detection 输出契约不同，拆成两个 typed catalog。
> - diffusion 4 个 sampler 通过同一个 base64 adapter 挂到通用 server，共用一个 catalog。
> - `test/model_catalog_unittest.cc` 校验字段完整性、全局唯一性、TOML section 与
>   model_config_file_path 存在性，并真实构造 `CvModelServer<OUTPUT>`。

## Phase 3：Scaffolder

- [x] 新增 `scripts/new_model.py`
- [x] 新增模型头文件模板
- [x] 新增模型实现模板
- [x] 新增 TOML 配置模板
- [x] 新增 output contract test 模板
- [x] 新增模型文档模板
- [x] 支持 `--dry-run`
- [x] 支持 `--force`
- [x] 支持 `--list-tasks`
- [x] 生成 catalog entry 提示
- [x] 生成 golden 占位提示
- [x] 脚手架自测
- [x] 生成后 tests-only 编译验证


> Phase 3 落地说明：
> - CLI 精简为 `--task/--name/--class/--backend/--dry-run/--force/--list-tasks/--check`，
>   去掉了原方案里的 `--family` 与 `--input`：输出契约来自 tasks.json，输入在 server 层固定。
> - 防漂移检查放在 `scripts/check_consistency.py` 而不是 C++ 测试里：它本质是对源码文本的
>   一致性校验，归 repo consistency checker 管，比在 gtest 里做字符串匹配更合适。
> - 脚手架绝不改共享文件（catalog / test CMake / golden test），只打印要粘贴的片段，
>   避免 `--force` 生成出人意料的 diff。
> - `src/models/object_detection/rtdetr_detector.*` 是留存的生成样例，同时充当
>   模板可编译性的长期哨兵；它不在任何 catalog 里，因此不可能被误启动。
> - 新增 `MODEL_NOT_IMPLEMENTED`（wire code 7）让未实现的模型显式失败而不是伪装成契约错误。

## Phase 4：IO 拆分

- [x] 拆分 common input
- [x] 拆分 classification IO
- [x] 拆分 object detection / face detection IO
- [x] 拆分 scene segmentation IO
- [x] 拆分 OCR IO
- [x] 拆分 matting IO
- [x] 拆分 enhancement IO
- [x] 拆分 feature point IO
- [x] 拆分 depth IO
- [x] 拆分 CLIP IO
- [x] 拆分 SAM IO
- [x] 拆分 diffusion IO
- [x] 保留 `model_io_define.h` 兼容聚合头
- [x] 验证编译依赖与编译时间变化


> Phase 4 落地说明：
> - 文件后缀跟随仓库习惯用 `.h`，不是原方案里的 `.hpp`。
> - **最大收益来自 `opencv2/opencv.hpp` → `opencv2/core.hpp`**，而不是拆目录本身：
>   IO 头的预处理行数 132,034 → 93,223（-29.5%），头文件 333 → 270（-19%），
>   只 include IO 头的叶子 TU 编译 2.23s → 1.25s（-44%，3 次取中位）。
> - `models` 库目标本身没有可测变化（18.6s → 19.2s，噪声范围内）：它的 .cpp
>   本来就拉 MNN/TRT 等更重的依赖。收益体现在只依赖 IO 类型的 TU 上。
> - 拆目录的即时价值是**所有权隔离**：改一个任务的 IO 类型不再需要动 254 行的
>   单体头；长期价值要等调用方逐步迁离聚合头后才会体现在增量编译上。
> - 聚合头 `model_io_define.h` 保留且 guard 不变，59 个调用方零改动；
>   `check_consistency.py` 禁止向聚合头回填类型定义，也禁止 IO 头使用 opencv.hpp。
> - 顺手修正了原文件里 `matting` namespace 错误的闭合注释（写成 scene_segmentation）。

## Phase 5：测试注册化

- [x] 新增 `test/model_contract_test_util.h`
- [x] contract test 自动生成 missing output 用例
- [x] 自动生成 wrong dtype 用例
- [x] 自动生成 wrong rank 用例
- [x] 自动生成 wrong shape 用例
- [x] 自动生成 short buffer 用例
- [x] 自动生成 NaN / Inf 用例
- [x] 新增 `test/model_golden_registry.h`
- [x] golden case 注册化
- [x] weights 缺失 skip 逻辑统一
- [x] golden 容差策略统一
- [x] 保持现有 golden 用例名称不变
- [x] 保持现有 golden 数据不变


> Phase 5 落地说明：
> - `test/model_golden_registry.h` 承载原有的全部 helper（权重检查 / 配置归一化 / 指纹 / 比对），
>   并提供 9 个按输出类型区分的 `GOLDEN_*_CASE` 宏，不做类型擦除。
> - `test/model_contract_test_util.h` 的 `POSTPROCESS_CONTRACT_TEST` 一行生成 7 个独立的
>   TEST（missing / wrong dtype / wrong rank / wrong shape / short buffer / NaN / Inf），
>   每个都能单独 `--gtest_filter`。
> - 21 个标准 golden 用例改为宏注册；6 个特殊用例（3 个 batch 一致性、SAM prompt/AMG、CLIP 双塔）
>   刻意保持手写，不为了统一而统一。
> - **零漂移硬校验**：迁移前后 27 个用例名与声明顺序完全一致，25 个 golden 基线文件 sha256
>   完全一致，golden 27/27 通过。
> - 脚手架 contract 模板从手写断言改为一行宏，canary 重新生成后 7/7 通过。
> - 一个刻意的范围缩减：宏不校验"输出未被部分污染"，因为这需要 OUTPUT 支持 operator==；
>   该语义留在各模型手写的 contract 测试里。

## Phase 6：模型族迁移

### 顺序清单

- [x] classification
- [x] matting
- [x] scene segmentation
- [x] enhancement
- [x] OCR
- [x] object detection
- [x] feature point
- [x] depth
- [x] FastSAM
- [x] CLIP
- [ ] diffusion

### 每个模型族验收

- [ ] 使用 `ImagePipeline` 替换普通 preprocess 样板
- [ ] 使用 `OutputReader` 替换手写 contract 样板
- [ ] 使用 `ParamReader` 替换手写参数解析
- [ ] 使用 `SessionIoValidator` 替换 session shape 检查
- [ ] 相关 contract tests 通过
- [ ] 相关 GPU golden 不漂移


> Phase 6 进度说明（截至 enhancement 批次）：
> - **已完成 4 / 11 个模型族**：classification（4 个模型，含 mobilenetv2 试点）、
>   matting（2）、enhancement（2/3，enlightengan 刻意保留手写）、OCR（1）。
> - 手写 `std::memcpy` 从 30 降到 17；`ImagePipeline` 使用从 3 升到 14。
> - 每个族都用 `scripts/golden_drift_check.py` 证明 27 个用例名与 25 个 golden
>   基线哈希零漂移。
> - **enlightengan 推迟**：双张量输出（input_src NCHW 3 通道 + input_gray NCHW 1 通道，
>   自定义 luma 公式）、16 对齐、alpha 提取——需要真正的 toolkit 扩展，不是模式替换。
> - **enhancement 前置条件分支**先行合入：`ImagePipeline::bgra_to_rgb()` +
>   realesrgan golden 改用彩色输入（原灰度图对通道顺序回归不敏感，负向验证证明
>   去掉转换后旧输入 PASSED、新输入 FAILED）。
> - **剩余**：scene segmentation、object detection、feature point、depth、FastSAM、
>   CLIP；diffusion 已建议移出本 phase 单独评估。

## Phase 7：多 Session 模板

- [ ] 新增 `src/models/backend/session_group.h`
- [ ] 支持 session 声明
- [ ] 支持 backend table 解析
- [ ] 支持 session 创建
- [ ] 支持 init 失败统一 reset
- [ ] SAM predictor 试点
- [ ] SAM AMG 试点
- [ ] LightGlue 试点
- [ ] CLIP 试点
- [ ] diffusion sampler / VAE 试点


> Phase 7 进度说明（步骤 1-3 完成）：
> - 新增 `src/models/backend/multi_session_model.h`：`IoSpec` / `SessionSpec` /
>   `MultiSessionModel<Derived, INPUT, OUTPUT>`，负责多引擎的创建、IO 校验与
>   失败统一清理；**刻意不做执行编排**，运行顺序与结果合并仍是模型自己的逻辑。
> - `create_session()` 是虚函数，测试可以注入 fake session，不需要真实模型文件。
> - `test/multi_session_model_unittest.cc` 4 个用例：正常创建 / IO 不匹配清空 /
>   引擎缺失清空 / 未声明名字返回 nullptr。
> - **CLIP 试点完成**：`OpenAiClip` 改继承 `MultiSessionModel`，删掉两个手写
>   `validate_*_io`、两个 session 成员指针和手写 create/reset 序列；CLIP golden
>   通过且零漂移。
> - 剩余：lightglue、SAM prompt、LDM、SAM AMG（AMG 的 8 个 decoder 需要先确认
>   是"多 session"还是"批量单 session"，不适合就明确保留手写）。


> **Phase 7 收尾结论（LDM 评估后）**：
>
> LDM **不迁移**，理由是结构性的，不是实现细节：
>
> 1. `LDMSampler` 继承 `BaseAiModel` 而非 `BackendCvModel`，而
>    `MultiSessionModel` 派生自 `BackendCvModel`；迁移要先改基类。
> 2. session 不是通过自身 section 的 `<key>_backend` 子表创建，而是从
>    **两个外部 TOML 文件**（`latent_diffusion_cfg` / `vae_decoder_cfg`）加载。
> 3. **决定性障碍**：DDIM 和 DDPM 是共享同一个 `shared_ptr<LatentDenoiseModel>`
>    的两个调度器——一个 session 被两条编排路径共用，与「每个命名引擎一个
>    session」正好相反。
> 4. 子模型（`DDPMUNet` / `AutoEncoderKL`）本身已是完整的 `BackendCvModel`，
>    各自通过自己的 `[DDPM_UNET].backend` / `[AUTOENCODER_KL].backend` 持有 session。
>
> LDM 是「多个独立模型的组合」，不是「一个多引擎模型」。强行套模板需要改基类、
> 重构配置加载、并破坏共享 UNet 的设计。且 LDM 没有 golden 用例，无零漂移兜底。
>
> **Phase 7 最终状态**：
> - ✅ CLIP、lightglue、SamPredictor 迁移到 `MultiSessionModel`
> - ⛔ SamAutoMaskGenerator（session 池 + Workflow 并发）、LDM（共享 UNet 的模型组合）
>   明确保留，理由均已记录
> - 两个「保留」案例说明 `MultiSessionModel` 的边界是清晰的：它解决的是
>   **固定数量、各自独立、按名索引**的多引擎生命周期，不是并发池也不是模型组合

## Phase 8：文档与引导

- [ ] 更新 `docs/how_to_add_new_model.md`
- [ ] 更新 `docs/how_to_add_new_model.zh-cn.md`
- [ ] 更新 `docs/model-contract-governance.md`
- [ ] 新增 10 分钟接入最小分类模型教程
- [ ] 新增 10 分钟接入 detection 模型教程
- [ ] 新增 output contract 编写指南
- [ ] 新增 golden 编写指南
- [ ] 新增 shape / dtype 排查指南
- [ ] 更新 README 双语入口

## 总验收

- [x] `scripts/check_consistency.py`
- [x] `scripts/gen_openapi.py --check`
- [x] clang-format
- [x] tests-only check
- [x] tests-only `-Werror`
- [x] CPU profile full check
- [x] full GPU `-Werror` model golden target build
- [x] 完整 GPU golden
- [x] TSAN
- [x] ASan / UBSan
- [ ] 新增普通模型接触点 <= 5
- [ ] 普通图像模型手写 `memcpy` 为 0
- [ ] 普通图像模型手写 CHW/HWC 打包为 0
- [ ] catalog 覆盖率 100%
- [ ] 脚手架生成后可直接编译
