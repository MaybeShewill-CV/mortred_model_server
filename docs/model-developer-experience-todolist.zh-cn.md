# P4 Modern Model Developer Experience TODO List

> 执行分支：`refactor/models-p4-developer-experience`  
> 方案文档：[model-developer-experience-p4.zh-cn.md](model-developer-experience-p4.zh-cn.md)  
> 勾选规则：只有该步骤代码、测试、文档和必要验证全部完成后才标记为完成。

## 总览

| 阶段 | 状态 |
|---|---|
| Phase 0 基线与度量 | 已完成 |
| Phase 1 Runtime Toolkit | 已完成 |
| Phase 2 Model Catalog | 未开始 |
| Phase 3 Scaffolder | 未开始 |
| Phase 4 IO 拆分 | 未开始 |
| Phase 5 测试注册化 | 未开始 |
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

- [ ] 设计任务内 `ModelEntry`
- [ ] 新增 `src/models/catalog/model_entry.h`
- [ ] 建立 classification catalog
- [ ] 建立 object detection catalog
- [ ] 建立 scene segmentation catalog
- [ ] 建立 OCR catalog
- [ ] 建立 matting catalog
- [ ] 建立 enhancement catalog
- [ ] 建立 feature point catalog
- [ ] 建立 depth catalog
- [ ] 建立 SAM catalog
- [ ] 建立 CLIP catalog
- [ ] 建立 diffusion catalog
- [ ] factory 从 catalog 创建模型
- [ ] server spec 从 catalog 驱动
- [ ] 新增 `test/model_catalog_unittest.cc`
- [ ] 校验 section 唯一性
- [ ] 校验配置文件存在
- [ ] 校证 catalog 覆盖现有 factory 暴露模型

## Phase 3：Scaffolder

- [ ] 新增 `scripts/new_model.py`
- [ ] 新增模型头文件模板
- [ ] 新增模型实现模板
- [ ] 新增 TOML 配置模板
- [ ] 新增 output contract test 模板
- [ ] 新增模型文档模板
- [ ] 支持 `--dry-run`
- [ ] 支持 `--force`
- [ ] 支持 `--list-tasks`
- [ ] 生成 catalog entry 提示
- [ ] 生成 golden 占位提示
- [ ] 脚手架自测
- [ ] 生成后 tests-only 编译验证

## Phase 4：IO 拆分

- [ ] 拆分 common input
- [ ] 拆分 classification IO
- [ ] 拆分 object detection / face detection IO
- [ ] 拆分 scene segmentation IO
- [ ] 拆分 OCR IO
- [ ] 拆分 matting IO
- [ ] 拆分 enhancement IO
- [ ] 拆分 feature point IO
- [ ] 拆分 depth IO
- [ ] 拆分 CLIP IO
- [ ] 拆分 SAM IO
- [ ] 拆分 diffusion IO
- [ ] 保留 `model_io_define.h` 兼容聚合头
- [ ] 验证编译依赖与编译时间变化

## Phase 5：测试注册化

- [ ] 新增 `test/model_contract_test_util.h`
- [ ] contract test 自动生成 missing output 用例
- [ ] 自动生成 wrong dtype 用例
- [ ] 自动生成 wrong rank 用例
- [ ] 自动生成 wrong shape 用例
- [ ] 自动生成 short buffer 用例
- [ ] 自动生成 NaN / Inf 用例
- [ ] 新增 `test/model_golden_registry.h`
- [ ] golden case 注册化
- [ ] weights 缺失 skip 逻辑统一
- [ ] golden 容差策略统一
- [ ] 保持现有 golden 用例名称不变
- [ ] 保持现有 golden 数据不变

## Phase 6：模型族迁移

### 顺序清单

- [ ] classification
- [ ] matting
- [ ] scene segmentation
- [ ] enhancement
- [ ] OCR
- [ ] object detection
- [ ] feature point
- [ ] depth
- [ ] FastSAM
- [ ] CLIP
- [ ] diffusion

### 每个模型族验收

- [ ] 使用 `ImagePipeline` 替换普通 preprocess 样板
- [ ] 使用 `OutputReader` 替换手写 contract 样板
- [ ] 使用 `ParamReader` 替换手写参数解析
- [ ] 使用 `SessionIoValidator` 替换 session shape 检查
- [ ] 相关 contract tests 通过
- [ ] 相关 GPU golden 不漂移

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
