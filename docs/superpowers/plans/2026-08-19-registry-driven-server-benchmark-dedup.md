# 注册表驱动的 server/benchmark 去重实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 把 22 个结构完全同构的具体 server 实现（`src/server/**/*_server.{h,cpp}`，各 193-205 行）收敛为一个泛型 `AiModelServer<OUTPUT>` 模板 + 每模型一份"注册表条目"（spec）；把 34 个 benchmark main 中可折叠的 29 个收敛为一个公共 `run_benchmark` 驱动 + 每模型一份 spec；同时把 22 个 apps/server main 薄壳化为公共入口。预计净削减约 8,500-9,000 行重复代码，且保持所有二进制名、CLI、HTTP 响应、配置格式不变。

**架构：** 服务侧：新增 `AiModelServer<MODEL_OUTPUT>`（持有 `AiServerSpec`：TOML 段名 ×2、worker 工厂、响应序列化器），工厂层用新加的 `TypeErasedFactory::register_creator` 注册"spec 闭包"替代 `register_type<具体类>`，随后删除 22 对 server 类文件；入口侧：新增 `run_model_server_main` 与 `run_benchmark` 两个公共 main 驱动，原 main 变成 10-25 行的 spec 声明。五个多阶段/有状态 benchmark（clip、sam×3、bytetrack）不属于"单模型循环"形状，明确保持原样。

**技术栈：** C++17 模板、toml11、rapidjson、GTest、CMake、Sogou Workflow；验证依赖本机 WSL 全量构建环境（见 `docs/wsl_build_test_report.md`，本机已有权重与 TRT engine，golden 测试可运行）。

---

## 迁移前实测基线（2026-08-19 盘点）

- C++ 源码总量：**259 文件 / 44,703 行**（`src + test`，含 .h/.inl/.cpp/.cc）。
- 22 个 server cpp 全部同构，唯一差异点已核实为 5 个：server TOML 段名、model TOML 段名、worker 工厂函数、输出类型、响应序列化函数（外加日志文案）。mobilenetv2 与 densenet 的 server 逐行 diff 仅有命名差异。
- 22 个 server 头文件的引用方**只有** 8 个 factory task 头文件（已全仓 grep 核实），删除它们的影响面封闭。
- `scripts/check_consistency.py` 硬约束：`conf/server/<category>` 必须对应 `src/apps/server/<category>` 目录且含 .cpp；`docs/repository-layout.md` 引用的每个 `src/...` 路径必须存在；openapi 文档与内嵌副本同步。→ **保留 22 个 app main 文件路径与全部 exe 名**。
- benchmark 34 个 main 分为三类：24 个"标准单图"（同构骨架：参数检查→读图→建模型→init→循环→日志→输出处理）、5 个"钩子可折叠"（lightglue 双图、diffusion×4 无图输入+额外 CLI 参数）、5 个"多阶段不折叠"（clip 四段循环、sam×3 encoder/predict 分阶段、bytetrack 帧目录+跟踪器状态）。
- 本机 WSL 已验证全量构建（71 exe + 4 so），权重/engine 就绪 → 全量编译与 golden 回归可在本机执行。

### server 注册表（22 条，迁移的完整数据源）

> "server 工厂函数"列 = factory 头文件中现有的 `create_X_server` 函数名（任务 3 改写其函数体、任务 4 薄壳 main 调用它）。名称已逐一从 `src/factory/*_task.h` 核实。

| # | 原 server（删除对象） | server TOML 段 | model TOML 段 | server 工厂函数 | worker 工厂（`<base64_input, OUTPUT>`） | OUTPUT | response 填充 |
|---|---|---|---|---|---|---|---|
| 1 | object_detection/yolov8_det_server | YOLOV8_DETECTION_SERVER | YOLOV8 | create_yolov8_det_server | create_yolov8_detector | std_object_detection_output | fill_object_detection |
| 2 | object_detection/yolov7_det_server | YOLOV7_DETECTION_SERVER | YOLOV7 | create_yolov7_det_server | create_yolov7_detector | std_object_detection_output | fill_object_detection |
| 3 | object_detection/yolov6_det_server | YOLOV6_DETECTION_SERVER | YOLOV6 | create_yolov6_det_server | create_yolov6_detector | std_object_detection_output | fill_object_detection |
| 4 | object_detection/yolov5_det_server | YOLOV5_DETECTION_SERVER | YOLOV5 | create_yolov5_det_server | create_yolov5_detector | std_object_detection_output | fill_object_detection |
| 5 | object_detection/nano_det_server | NANODET_DETECTION_SERVER | NANODET | create_nanodet_det_server | create_nanodet_detector | std_object_detection_output | fill_object_detection |
| 6 | object_detection/centerface_det_server | CENTER_FACE_DETECTION_SERVER | CENTER_FACE | create_centerface_det_server | create_centerface_detector | std_face_detection_output | fill_face_detection |
| 7 | object_detection/libface_det_server | LIBFACE_DETECTION_SERVER | LIBFACE | create_libface_det_server | create_libface_detector | std_face_detection_output | fill_face_detection |
| 8 | classification/densenet_server | DENSENET_CLASSIFICATION_SERVER | DENSENET | create_densenet_cls_server | create_densenet_classifier | std_classification_output | fill_classification |
| 9 | classification/mobilenetv2_server | MOBILENETV2_CLASSIFICATION_SERVER | MOBILENETV2 | create_mobilenetv2_cls_server | create_mobilenetv2_classifier | std_classification_output | fill_classification |
| 10 | classification/resnet_server | RESNET_CLASSIFICATION_SERVER | RESNET | create_resnet_cls_server | create_resnet_classifier | std_classification_output | fill_classification |
| 11 | scene_segmentation/bisenetv2_server | BISENETV2_SERVER | BISENETV2 | create_bisenetv2_server | create_bisenetv2_segmentor | std_scene_segmentation_output | fill_scene_segmentation |
| 12 | scene_segmentation/hrnet_server | HRNET_SERVER | HRNET | create_hrnet_server | create_hrnet_segmentor | std_scene_segmentation_output | fill_scene_segmentation |
| 13 | scene_segmentation/pphuman_seg_server | PPHUMAN_SEG_SERVER | PPHUMAN_SEG | create_pphuman_seg_server | create_pphuman_segmentor | std_scene_segmentation_output | fill_scene_segmentation |
| 14 | matting/modnet_server | MODNET_SERVER | MODNET | create_modnet_server | create_modnet_segmentor | std_matting_output | fill_matting |
| 15 | matting/pp_matting_server | PP_MATTING_SERVER | PP_MATTING | create_pp_matting_server | create_ppmatting_segmentor | std_matting_output | fill_matting |
| 16 | enhancement/attentive_gan_derain_server | ATTENTIVE_GAN_DERAIN_SERVER | ATTENTIVE_GAN_DERAIN | create_attentivegan_derain_server | create_attentivegan_enhancementor | std_enhancement_output | fill_enhancement |
| 17 | enhancement/enlighten_gan_server | ENLIGHTEN_GAN_SERVER | ENLIGHTEN_GAN | create_enlightengan_server | create_enlightengan_enhancementor | std_enhancement_output | fill_enhancement |
| 18 | enhancement/real_esr_gan_server | REAL_ESRGAN_SERVER | REAL_ESRGAN | create_realesrgan_server | create_realesrgan_enhancementor | std_enhancement_output | fill_enhancement |
| 19 | mono_depth_estimation/depth_anything_server | DEPTH_ANYTHING_ESTIMATION_SERVER | DEPTH_ANYTHING | create_depth_anything_estimation_server | create_depth_anything_estimator | std_mde_output | fill_depth_estimation |
| 20 | mono_depth_estimation/metric3d_server | METRIC3D_ESTIMATION_SERVER | METRIC3D | create_metric3d_estimation_server | create_metric3d_estimator | std_mde_output | fill_depth_estimation |
| 21 | ocr/dbnet_server | DBNET_SERVER | DBNET | create_dbtext_detection_server | create_dbtext_detector | std_text_regions_output | fill_text_regions |
| 22 | feature_point/superpoint_fp_server | SUPERPOINT_FP_SERVER | SUPERPOINT | create_superpoint_fp_server | create_superpoint_extractor | std_feature_point_output | fill_feature_points |

> app main 对应的 `server TOML 段` 同时是薄壳 main 读取 `host/port` 的段名（见任务 4）。

### benchmark 注册表（29 条折叠 + 5 条保留）

| 分组 | 文件（src/apps/model_benchmark/ 下） | 建模方式 | 循环数 | 输出处理 |
|---|---|---|---|---|
| 分类×4 | classification/{densenet,dinov2,mobilenetv2,resnet}_benchmark | 标准单图，默认图 `classification/ILSVRC2012_val_00000003.JPEG` | 1000/100/1000/1000 | log_classification |
| 检测×7 | object_detection/{yolov5..8,nanodet,centerface,libface}_benchmark | 标准单图，默认图见原文件 | 100 | vis_object_detection(cls_num)+保存 |
| 分割×4 | segmentation/{bisenetv2,hrnet,msocrnet,pphumanseg}_benchmark | 标准单图 | 100/10/100/500 | colorize_segmentation_mask+保存 |
| 抠图×2 | matting/{modnet,ppmatting}_benchmark | 标准单图 | 100 | 保存 matting_result |
| 增强×3 | enhancement/{attentivegan,enlightengan,real_esrgan}_benchmark | 标准单图（默认图在 low_light 等子目录，照搬原文件） | 100 | 保存 enhancement_result |
| 深度×2 | mono_depth_estimation/{depth_anything,metric3d}_benchmark | 标准单图 | 10 | 保存 colorized_depth_map（depth_anything 另存 yaml，照搬原尾） |
| OCR×1 | ocr/dbnet_benchmark | 标准单图 | 100 | vis_text_detection+保存 |
| 特征点×1 | feature_point/superpoint_benchmark | 标准单图 | 100 | vis_feature_points(img,out,4)+保存 |
| 双图×1 | feature_point/lightglue_benchmark | make_input 钩子（pair_mat_input，两幅默认图） | 100 | 照搬原尾保存逻辑 |
| 扩散×4 | diffusion/{ddpm,ddim,cls_cond_ddim,ldm}_sampler 系 | make_input 钩子（sample_size/steps 等 CLI 参数），直接 `make_unique` 建模 | 1 | 保存采样图（照搬原尾） |
| **保留×5** | clip/openai_clip、sam/{sam,fast_sam,sam_amg}、mot/bytetrack | **不折叠**：clip 是 4 段独立循环；sam 是 encoder 计时+predict 演示两阶段；bytetrack 是帧目录遍历+跟踪器跨帧状态。形状不同，强行套模板会扭曲设计 | - | - |

---

## 文件结构

| 操作 | 文件 | 职责 |
|---|---|---|
| 修改 | `src/factory/base_factory.h` | 新增 `register_creator(name, closure)`（向后兼容，不改 `register_type`） |
| 修改 | `test/model_factory_unittest.cc` | 追加 register_creator 单测（tests-only 可跑，TDD） |
| 创建 | `src/server/generic_ai_server.h` | `AiServerSpec<OUTPUT>` + `AiModelServer<OUTPUT>` 泛型服务（唯一 init/fill 实现） |
| 删除 | `src/server/{8 个任务目录}/...` 的 22 对 `*_server.{h,cpp}` | 被泛型模板 + 注册表条目取代 |
| 修改 | `src/factory/{8 个 task 头文件}` | `create_X_server` 改为注册 spec 闭包；移除对 22 个 server 头的 include |
| 修改 | `src/server/CMakeLists.txt` | 移除 22 个 cpp 条目，加入 generic_ai_server.h（库变纯头文件载体，与 factory 现状一致） |
| 创建 | `src/apps/common/model_server_main.h` | `run_model_server_main` 公共服务入口（glog 初始化/TOML 解析/start/wait） |
| 修改 | `src/apps/server/**/` 22 个 main | 薄壳化为一行调用（文件路径、exe 名全部不变） |
| 创建 | `src/apps/common/benchmark_runner.h` | `BenchmarkSpec` + `run_benchmark` 驱动（预热+mean/p50/p99）+ 常用输入/输出 handler |
| 修改 | `src/apps/model_benchmark/**/` 29 个 main | 薄壳化为 spec 声明（5 个多阶段 benchmark 保持原样） |
| 修改 | `src/apps/CMakeLists.txt`、`docs/repository-layout.md`、`docs/how_to_add_new_server.md` | 记录新文件/新扩展流程 |

---

### 任务 1：`TypeErasedFactory::register_creator` + 单测（TDD，tests-only 可验证）

**文件：**
- 修改：`test/model_factory_unittest.cc`
- 修改：`src/factory/base_factory.h`

- [ ] **步骤 1：编写失败的测试**

在 `test/model_factory_unittest.cc` 末尾追加（沿用该文件既有的 Base/Drived 测试夹具风格）：

```cpp
TEST(TypeErasedFactory, RegisterCreatorBuildsCustomClosure) {
    struct CustomBase { virtual ~CustomBase() = default; virtual int value() const = 0; };
    struct CustomDerived : CustomBase { int value() const override { return 42; } };

    auto& factory = jinq::factory::TypeErasedFactory<CustomBase>::get_instance();
    factory.register_creator("custom_closure", []() -> std::unique_ptr<CustomBase> {
        return std::unique_ptr<CustomBase>(new CustomDerived());
    });
    auto obj = factory.create("custom_closure");
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->value(), 42);
}

TEST(TypeErasedFactory, RegisterCreatorRejectsEmptyNameOrNullClosure) {
    struct EmptyBase { virtual ~EmptyBase() = default; };
    auto& factory = jinq::factory::TypeErasedFactory<EmptyBase>::get_instance();
    factory.register_creator("", []() -> std::unique_ptr<EmptyBase> { return nullptr; });
    EXPECT_EQ(factory.create(""), nullptr);
    factory.register_creator("null_closure", nullptr);
    EXPECT_EQ(factory.create("null_closure"), nullptr);
}
```

- [ ] **步骤 2：运行测试验证失败**

```bash
cmake --preset tests-only && cmake --build --preset tests-only
ctest --preset tests-only -R model_factory_unittest -V
```

预期：编译失败，`register_creator` 不是 `TypeErasedFactory` 的成员。

- [ ] **步骤 3：实现 `register_creator`**

在 `src/factory/base_factory.h` 的 `register_type` 之后新增（与 `register_type` 共用 `creator_t` 与互斥量）：

```cpp
    /***
     * Register an arbitrary creator closure (used by spec-driven servers whose
     * concrete type is a template instantiation, not a named class). Same
     * overwrite-on-same-name and mutex semantics as register_type.
     */
    void register_creator(const std::string& name, creator_t creator) {
        if (name.empty() || !creator) {
            LOG(ERROR) << "refusing to register a null creator or a creator with an empty name";
            return;
        }
        std::lock_guard<std::mutex> lock(_m_mutex);
        _m_creators[name] = std::move(creator);
    }
```

注意：`creator_t` 的 using 声明目前在类私有区末尾，需把 `using creator_t = ...` 上移到 `register_creator` 之前（或放到类开头公有区），保持 `register_type` 不变。

- [ ] **步骤 4：运行测试验证通过**

```bash
cmake --build --preset tests-only && ctest --preset tests-only -R model_factory_unittest -V
```

预期：PASS（新旧用例全绿）。

- [ ] **步骤 5：Commit**

```bash
git add src/factory/base_factory.h test/model_factory_unittest.cc
git commit -m "feat(factory): support creator closures in TypeErasedFactory"
```

---

### 任务 2：`AiModelServer<OUTPUT>` 泛型模板 + YoloV8 试点迁移

**文件：**
- 创建：`src/server/generic_ai_server.h`
- 删除：`src/server/object_detection/yolov8_det_server.h`、`src/server/object_detection/yolov8_det_server.cpp`
- 修改：`src/factory/obj_detection_task.h`（yolov8 的 create 函数 + include）
- 修改：`src/server/CMakeLists.txt`

- [ ] **步骤 1：编写 `src/server/generic_ai_server.h`**

```cpp
/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * File: generic_ai_server.h
 * Date: 2026-08-19
 *
 * Registry-driven generic model server: the single implementation that all
 * 22 former hand-written concrete servers delegate to. Per-model variation
 * lives in AiServerSpec (TOML sections, worker factory, response filler).
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_GENERIC_AI_SERVER_H
#define MORTRED_MODEL_SERVER_GENERIC_AI_SERVER_H

#include <functional>
#include <memory>
#include <string>

#include "toml/toml.hpp"
#include "rapidjson/document.h"
#include "workflow/WFHttpServer.h"

#include "common/file_path_util.h"
#include "common/status_code.h"
#include "models/base_model.h"
#include "models/model_io_define.h"
#include "server/abstract_server.h"
#include "server/base_server_impl.h"

namespace jinq {
namespace server {

using Base64Input = jinq::models::io_define::common_io::base64_input;

template<typename MODEL_OUTPUT>
using AiWorkerPtr = std::unique_ptr<jinq::models::BaseAiModel<Base64Input, MODEL_OUTPUT>>;

template<typename MODEL_OUTPUT>
using AiWorkerFactory = std::function<AiWorkerPtr<MODEL_OUTPUT>(const std::string&)>;

template<typename MODEL_OUTPUT>
using AiResponseFiller = void (*)(rapidjson::Document::AllocatorType&,
                                  rapidjson::Document&,
                                  const MODEL_OUTPUT&);

/***
 * Per-model server registration entry. This is the whole per-model footprint
 * that used to be a 200-line Impl copy.
 */
template<typename MODEL_OUTPUT>
struct AiServerSpec {
    std::string server_section;        // e.g. "YOLOV8_DETECTION_SERVER"
    std::string model_section;         // e.g. "YOLOV8" (holds model_config_file_path)
    std::string display_name;          // e.g. "Yolov8 object detection"
    AiWorkerFactory<MODEL_OUTPUT> make_worker;
    AiResponseFiller<MODEL_OUTPUT> fill_response;
};

template<typename MODEL_OUTPUT>
class AiModelServer final : public BaseAiServer {
  public:
    explicit AiModelServer(AiServerSpec<MODEL_OUTPUT> spec)
        : _m_spec(std::move(spec)), _m_impl(std::make_unique<Impl>(_m_spec)) {}

    AiModelServer(const AiModelServer&) = delete;
    AiModelServer& operator=(const AiModelServer&) = delete;

    jinq::common::StatusCode init(const toml::table& config) override {
        auto status = _m_impl->init(config);
        if (status != jinq::common::StatusCode::OK) {
            LOG(INFO) << "init " << _m_spec.display_name << " server failed";
            return status;
        }
        return init_http_server(_m_impl.get());
    }

    void serve_process(WFHttpTask* task) override {
        _m_impl->serve_process(task);
    }

    bool is_successfully_initialized() const override {
        return _m_impl->is_successfully_initialized();
    }

  private:
    class Impl : public BaseAiServerImpl<AiWorkerPtr<MODEL_OUTPUT>, MODEL_OUTPUT> {
      public:
        explicit Impl(const AiServerSpec<MODEL_OUTPUT>& spec) : _m_spec(spec) {}

        jinq::common::StatusCode init(const toml::table& config) override;

        void fill_response_data(rapidjson::Document::AllocatorType& allocator,
                                rapidjson::Document& data,
                                const jinq::common::StatusCode& status,
                                const MODEL_OUTPUT& model_output) override {
            (void)status;  // 契约：仅成功路径调用
            _m_spec.fill_response(allocator, data, model_output);
        }

      private:
        const AiServerSpec<MODEL_OUTPUT>& _m_spec;
    };

    AiServerSpec<MODEL_OUTPUT> _m_spec;
    std::unique_ptr<Impl> _m_impl;
};

/*********** Public Func Sets **************/

template<typename MODEL_OUTPUT>
jinq::common::StatusCode AiModelServer<MODEL_OUTPUT>::Impl::init(const toml::table& config) {
    const toml::table* server_section_ptr = config[_m_spec.server_section].as_table();
    if (server_section_ptr == nullptr) {
        LOG(ERROR) << "Config section " << _m_spec.server_section << " missing or not a table";
        _m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    const toml::table& server_section = *server_section_ptr;

    auto common_status = parse_common_server_config(server_section);
    if (common_status != jinq::common::StatusCode::OK) {
        return common_status;
    }
    auto worker_nums = parse_worker_nums(server_section);
    if (worker_nums <= 0) {
        _m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }

    const toml::table* model_section_ptr = config[_m_spec.model_section].as_table();
    if (model_section_ptr == nullptr) {
        LOG(ERROR) << "Config section " << _m_spec.model_section << " missing or not a table";
        _m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg_path = (*model_section_ptr)["model_config_file_path"].value_or<std::string>("");
    if (!jinq::common::FilePathUtil::is_file_exist(model_cfg_path)) {
        LOG(ERROR) << _m_spec.display_name << " model config file not exist: " << model_cfg_path;
        _m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }

    auto model_cfg_parsed = toml::parse_file(model_cfg_path);
    if (!model_cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(model_cfg_parsed.error().description());
        _m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    auto model_cfg = std::move(model_cfg_parsed).table();

    for (int index = 0; index < worker_nums; ++index) {
        auto worker = _m_spec.make_worker("worker_" + std::to_string(index + 1));
        if (!worker->is_successfully_initialized()) {
            if (worker->init(model_cfg) != jinq::common::StatusCode::OK) {
                _m_successfully_initialized = false;
                return jinq::common::StatusCode::SERVER_INIT_FAILED;
            }
        }
        _m_working_queue.enqueue(std::move(worker));
    }

    if (!server_section.contains("server_uri")) {
        LOG(ERROR) << "missing server uri field";
        _m_successfully_initialized = false;
        return jinq::common::StatusCode::SERVER_INIT_FAILED;
    }
    _m_server_uri = server_section["server_uri"].value_or<std::string>("");

    // commit the worker watermark only after the queue is fully filled
    _m_worker_nums = static_cast<size_t>(worker_nums);
    _m_successfully_initialized = true;
    LOG(INFO) << _m_spec.display_name << " server init successfully";
    return jinq::common::StatusCode::OK;
}

}  // namespace server
}  // namespace jinq

#endif  // MORTRED_MODEL_SERVER_GENERIC_AI_SERVER_H
```

实现说明：`init` 主体是 22 个现有实现的逐字参数化移植（以 `src/server/object_detection/yolov8_det_server.cpp` 为基准），行为不变——同样的校验顺序、同样的错误码、同样的水位提交时机。

- [ ] **步骤 2：改写工厂注册（obj_detection_task.h 的 yolov8 条目）**

在 `src/factory/obj_detection_task.h` 中：
1. 删除 `#include "server/object_detection/yolov8_det_server.h"`；
2. 头部补充 `#include "server/generic_ai_server.h"`、`#include "server/response_serializers.h"`（若尚未包含）；
3. 将 `create_yolov8_det_server` 整个函数替换为：

```cpp
// create yolov8 object detection server
inline std::unique_ptr<BaseAiServer> create_yolov8_det_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        AiServerSpec<jinq::models::io_define::object_detection::std_object_detection_output> spec;
        spec.server_section = "YOLOV8_DETECTION_SERVER";
        spec.model_section = "YOLOV8";
        spec.display_name = "Yolov8 object detection";
        spec.make_worker = [](const std::string& name) {
            return create_yolov8_detector<
                jinq::server::Base64Input,
                jinq::models::io_define::object_detection::std_object_detection_output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_object_detection;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<
                jinq::models::io_define::object_detection::std_object_detection_output>(
                std::move(spec)));
    });
    return server_factory.create(server_name);
}
```

（该文件内如已有 `using jinq::models::io_define::object_detection::std_object_detection_output;` 等别名，可用短名，保持与文件既有风格一致。）

- [ ] **步骤 3：删除旧文件并更新构建**

```bash
git rm src/server/object_detection/yolov8_det_server.h src/server/object_detection/yolov8_det_server.cpp
```

`src/server/CMakeLists.txt`：从 `SERVER_LIB_SRC` 删除 `${CMAKE_CURRENT_LIST_DIR}/object_detection/yolov8_det_server.cpp` 一行，新增 `${CMAKE_CURRENT_LIST_DIR}/generic_ai_server.h`。

- [ ] **步骤 4：WSL 全量编译 + 契约/golden 回归**

```bash
# WSL 内（沿用 docs/wsl_build_test_report.md 的构建目录）
cmake -B build -DMORTRED_BUILD_FULL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16
ctest --test-dir build -R "server_e2e_contract_test|response_schema_test|openapi_consistency_test" --output-on-failure
ctest --test-dir build -R model_golden_test --output-on-failure
python3 scripts/check_consistency.py
```

预期：全部通过；`model_golden_test` 中 yolov5-8 检测用例无指纹漂移（服务层改动不影响模型输出，golden 是行为锚点）。

- [ ] **步骤 5：冒烟验证（HTTP 行为不变）**

```bash
cd build && ./_bin/yolov8_detection_server.out ../conf/server/object_detection/yolov8/yolov8_server_config.toml &
curl -s http://127.0.0.1:9006/healthz    # 端口以该 toml 为准
curl -s http://127.0.0.1:9006/openapi.json | head -c 200
cd ../scripts && PYTHONPATH=$PWD python3 server/test_server.py --server yolov8 --mode single
```

预期：与迁移前相同的响应包络（`code/msg/data`）、相同的检测 JSON 字段。

- [ ] **步骤 6：Commit**

```bash
git add src/server/generic_ai_server.h src/server/CMakeLists.txt src/factory/obj_detection_task.h
git commit -m "refactor(server): introduce registry-driven AiModelServer, migrate yolov8"
```

---

### 任务 3：其余 21 个 server 迁移（按任务类别分 8 个 commit）

**文件：** 上方"server 注册表"表 2-22 行对应的 `*_server.{h,cpp}`（删除）、8 个 factory task 头文件（修改）、`src/server/CMakeLists.txt`（修改）。

每个类别的操作流程（跨类别通用，配合注册表行的数据即可独立执行）：

1. 在对应 factory task 头文件中删除 `#include "server/<分类>/<旧类头>.h"`；
2. 把该模型的 `create_X_server` 函数替换为 spec 闭包注册（模板见下）；
3. `git rm` 该模型的旧 `*_server.{h,cpp}`，并从 `src/server/CMakeLists.txt` 移除对应 cpp 行；
4. `cmake --build build -j16 && ctest --test-dir build -R "server_e2e_contract_test|model_golden_test" --output-on-failure`。

spec 闭包模板（以分类族 densenet 为例，注册表第 8 行数据；替换 `src/factory/classification_task.h` 中的 `create_densenet_cls_server`）：

```cpp
// create densenet classification server
inline std::unique_ptr<BaseAiServer> create_densenet_cls_server(const std::string& server_name) {
    auto& server_factory = ServerFactory<BaseAiServer>::get_instance();
    server_factory.register_creator(server_name, []() -> std::unique_ptr<BaseAiServer> {
        AiServerSpec<jinq::models::io_define::classification::std_classification_output> spec;
        spec.server_section = "DENSENET_CLASSIFICATION_SERVER";
        spec.model_section = "DENSENET";
        spec.display_name = "Densenet classification";
        spec.make_worker = [](const std::string& name) {
            return create_densenet_classifier<
                jinq::server::Base64Input,
                jinq::models::io_define::classification::std_classification_output>(name);
        };
        spec.fill_response = &jinq::server::response::fill_classification;
        return std::unique_ptr<BaseAiServer>(
            new jinq::server::AiModelServer<
                jinq::models::io_define::classification::std_classification_output>(
                std::move(spec)));
    });
    return server_factory.create(server_name);
}
```

提交切分（每类一次 commit，独立可评审、可回滚）：

- [ ] **3.1 目标检测其余 6 个**（yolov5/6/7、nanodet→std_object_detection_output+fill_object_detection；centerface/libface→std_face_detection_output+fill_face_detection）。Commit: `refactor(server): migrate remaining object detection servers to registry`
- [ ] **3.2 分类 3 个**（densenet/mobilenetv2/resnet→std_classification_output+fill_classification）。Commit: `refactor(server): migrate classification servers to registry`
- [ ] **3.3 场景分割 3 个**（bisenetv2/hrnet/pphuman_seg→std_scene_segmentation_output+fill_scene_segmentation）。Commit: `refactor(server): migrate scene segmentation servers to registry`
- [ ] **3.4 抠图 2 个**（modnet/pp_matting→std_matting_output+fill_matting）。Commit: `refactor(server): migrate matting servers to registry`
- [ ] **3.5 增强 3 个**（attentive_gan/enlighten_gan/real_esr_gan→std_enhancement_output+fill_enhancement）。Commit: `refactor(server): migrate enhancement servers to registry`
- [ ] **3.6 单目深度 2 个**（depth_anything/metric3d→std_mde_output+fill_depth_estimation）。Commit: `refactor(server): migrate depth estimation servers to registry`
- [ ] **3.7 OCR 1 个**（dbnet→std_text_regions_output+fill_text_regions）。Commit: `refactor(server): migrate dbnet server to registry`
- [ ] **3.8 特征点 1 个**（superpoint→std_feature_point_output+fill_feature_points）。Commit: `refactor(server): migrate superpoint server to registry`

每个类别完成后立即运行任务 2 步骤 4 的回归命令；全部完成后额外执行：

```bash
# 确认旧类无残留引用
grep -rn "DetServer\b\|ClassificationServer\b\|SegServer\b\|MattingServer\b\|EnhancementServer\|EstimationServer\|FPService\|DbnetServer" src --include=*.h --include=*.cpp | grep -v generic_ai_server
```

预期：无输出（factory 头文件的 using 声明与 include 全部清理干净）。

---

### 任务 4：22 个 apps/server main 薄壳化

**文件：**
- 创建：`src/apps/common/model_server_main.h`
- 修改：`src/apps/server/{8 个任务目录}/` 下全部 22 个 main cpp（路径不变）
- 修改：`src/apps/CMakeLists.txt`（无需改目标，只需确认 include 路径传播）

- [ ] **步骤 1：编写 `src/apps/common/model_server_main.h`**

```cpp
/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * File: model_server_main.h
 * Date: 2026-08-19
 *
 * Shared main() body for all model server executables.
 ************************************************/
#ifndef MORTRED_APPS_MODEL_SERVER_MAIN_H
#define MORTRED_APPS_MODEL_SERVER_MAIN_H

#include <functional>
#include <memory>
#include <string>

#include <glog/logging.h>
#include <workflow/WFFacilities.h>
#include "toml/toml.hpp"

#include "server/abstract_server.h"

namespace jinq {
namespace apps {

inline int run_model_server_main(
    int argc, char** argv,
    const std::string& server_section,
    const std::function<std::unique_ptr<jinq::server::BaseAiServer>(const std::string&)>& make_server) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    if (argc != 2) {
        LOG(INFO) << "usage:";
        LOG(INFO) << "exe cfg_path";
        return -1;
    }

    WFFacilities::WaitGroup wait_group(1);

    std::string config_file_path = argv[1];
    LOG(INFO) << "cfg file path: " << config_file_path;
    auto config_parsed = toml::parse_file(config_file_path);
    if (!config_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(config_parsed.error().description());
        return -1;
    }
    auto config = std::move(config_parsed).table();
    const auto& server_cfg = config[server_section];
    auto port = server_cfg["port"].value_or<int64_t>(0);
    auto host = server_cfg["host"].value_or<std::string>("127.0.0.1");
    LOG(INFO) << "serve on port: " << port;

    auto server = make_server("server");
    auto status = server->init(config);
    if (status != jinq::common::StatusCode::OK) {
        LOG(ERROR) << "server init failed, status: "
                   << std::to_string(static_cast<int>(status));
        return -1;
    }
    if (server->start(host.c_str(), static_cast<unsigned short>(port)) == 0) {
        wait_group.wait();
        server->stop();
        return 0;
    }
    LOG(ERROR) << "Cannot start server";
    return -1;
}

}  // namespace apps
}  // namespace jinq

#endif  // MORTRED_APPS_MODEL_SERVER_MAIN_H
```

（主体为 `src/apps/server/object_detection/yolov5_detection_server.cpp` 的逐字提取；唯一泛化点是 server_section 参数与 make_server 回调。原实现里工厂注册名如 `"yolov5_det_server"` 无外部消费者——ServerFactory 是进程内注册表——统一为 `"server"` 不影响任何行为。）

- [ ] **步骤 2：逐个替换 22 个 main（以 yolov8 为例，其余同理）**

`src/apps/server/object_detection/yolov8_detection_server.cpp` 全文替换为：

```cpp
// yolov8 detection server tool

#include "apps/common/model_server_main.h"
#include "factory/obj_detection_task.h"

int main(int argc, char** argv) {
    return jinq::apps::run_model_server_main(
        argc, argv, "YOLOV8_DETECTION_SERVER",
        [](const std::string& name) {
            return jinq::factory::object_detection::create_yolov8_det_server(name);
        });
}
```

其余 21 个 main 按"server 注册表"的 server TOML 段列替换段名与 create 函数；文件路径、文件名、生成的 exe 名一律不变（`check_consistency.py` 的目录映射与 `docs/repository-layout.md` 的产物表因此保持有效）。

- [ ] **步骤 3：构建 + 冒烟**

```bash
cmake --build build -j16
cd build && ./_bin/mobilenetv2_classification_server.out \
  ../conf/server/classification/mobilenetv2/mobilenetv2_server_config.toml &
PYTHONPATH=../scripts python3 ../scripts/server/test_server.py --server mobilenetv2 --mode single
```

预期：服务正常启动、客户端 1000 次请求全部成功（与 README Quick Start 流程一致）。

- [ ] **步骤 4：Commit（与任务 3 同类别切分）**

```bash
git add src/apps/common/model_server_main.h src/apps/server
git commit -m "refactor(apps): thin wrappers for model server mains"
```

---

### 任务 5：`benchmark_runner` 公共驱动 + mobilenetv2 试点

**文件：**
- 创建：`src/apps/common/benchmark_runner.h`
- 修改：`src/apps/model_benchmark/classification/mobilenetv2_benchmark.cpp`

- [ ] **步骤 1：编写 `src/apps/common/benchmark_runner.h`**

```cpp
/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * File: benchmark_runner.h
 * Date: 2026-08-19
 *
 * Shared benchmark main() driver: arg/config handling, model init, warmup,
 * timed loop with mean/p50/p99, and per-task output hooks.
 ************************************************/
#ifndef MORTRED_APPS_BENCHMARK_RUNNER_H
#define MORTRED_APPS_BENCHMARK_RUNNER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "toml/toml.hpp"

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/base_model.h"
#include "models/model_io_define.h"

namespace jinq {
namespace apps {

template<typename INPUT, typename OUTPUT>
struct BenchmarkSpec {
    std::string model_name;        // factory registration key (informational)
    std::string display_name;      // e.g. "mobilenetv2 classifier"
    std::string usage;             // e.g. "exe config_file_path [test_image_path]"
    int loops = 100;
    // 返回 false 时驱动打印 usage 并退出：标准 main 用 standard_args_ok
    std::function<bool(int argc)> args_ok;
    // 从 CLI + config 构造模型输入；标准单图 main 用 make_single_image_input
    std::function<INPUT(int argc, char** argv, const toml::table& cfg)> make_input;
    // 输出命名/可视化所需的输入图路径（无图 benchmark 返回 ""）
    std::function<std::string(int argc, char** argv)> image_path_of;
    // 输入有效性校验：标准单图 spec 检查 !input_image.empty()；
    // diffusion/lightglue 等自定义输入返回 true（由 make_input 自行保证）
    std::function<bool(const INPUT&)> input_ok;
    // 建模：走工厂 create_X 或直接 make_unique（diffusion 族）
    std::function<std::unique_ptr<jinq::models::BaseAiModel<INPUT, OUTPUT>>(const std::string&)> make_model;
    // 循环后调用一次：记日志 / 可视化 / 保存
    std::function<void(const INPUT& in, const OUTPUT& out, const std::string& image_path)> handle_output;
};

inline bool standard_args_ok(int argc) {
    return argc == 2 || argc == 3;
}

inline std::string standard_image_path(int argc, char** argv, const std::string& default_path) {
    if (argc >= 3) {
        LOG(INFO) << "input test image path: " << argv[2];
        return argv[2];
    }
    LOG(INFO) << "use default input test image path: " << default_path;
    return default_path;
}

inline jinq::models::io_define::common_io::mat_input make_single_image_input(
    int argc, char** argv, const std::string& default_path, int imread_flags = cv::IMREAD_COLOR) {
    const std::string path = standard_image_path(argc, argv, default_path);
    if (!jinq::common::FilePathUtil::is_file_exist(path)) {
        LOG(INFO) << "test input image file: " << path << " not exist";
        return jinq::models::io_define::common_io::mat_input{};
    }
    jinq::models::io_define::common_io::mat_input input;
    input.input_image = cv::imread(path, imread_flags);
    return input;
}

template<typename INPUT, typename OUTPUT>
int run_benchmark(int argc, char** argv, const BenchmarkSpec<INPUT, OUTPUT>& spec) {
    if (!spec.args_ok(argc)) {
        LOG(ERROR) << "wrong usage";
        LOG(INFO) << spec.usage;
        return -1;
    }
    const std::string cfg_file_path = argv[1];
    LOG(INFO) << "config file path: " << cfg_file_path;
    if (!jinq::common::FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(INFO) << "config file: " << cfg_file_path << " not exist";
        return -1;
    }
    auto cfg_parsed = toml::parse_file(cfg_file_path);
    if (!cfg_parsed) {
        LOG(ERROR) << "parse toml config file failed, error: "
                   << std::string(cfg_parsed.error().description());
        return -1;
    }
    auto cfg = std::move(cfg_parsed).table();

    INPUT model_input{};
    OUTPUT model_output{};
    model_input = spec.make_input(argc, argv, cfg);
    // 与原实现一致：输入缺失/解码失败时直接退出（标准单图检查空图）
    if (!spec.input_ok(model_input)) {
        return -1;
    }

    auto model = spec.make_model(spec.model_name);
    model->init(cfg);
    if (!model->is_successfully_initialized()) {
        LOG(INFO) << spec.display_name << " init failed";
        return -1;
    }

    LOG(INFO) << "start " << spec.display_name << " benchmark at: "
              << jinq::common::Timestamp::now().to_format_str();

    // warmup run: not counted (fixes the "first iteration skew" noted in review)
    model->run(model_input, model_output);

    std::vector<double> iter_ms(static_cast<size_t>(spec.loops));
    auto ts = jinq::common::Timestamp::now();
    for (int i = 0; i < spec.loops; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        model->run(model_input, model_output);
        const auto t1 = std::chrono::steady_clock::now();
        iter_ms[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    auto cost_time = jinq::common::Timestamp::now() - ts;

    std::sort(iter_ms.begin(), iter_ms.end());
    const double mean_ms = std::accumulate(iter_ms.begin(), iter_ms.end(), 0.0) / iter_ms.size();
    const double p50_ms = iter_ms[iter_ms.size() / 2];
    const double p99_ms = iter_ms[static_cast<size_t>(
        std::ceil(0.99 * static_cast<double>(iter_ms.size())) - 1)];
    LOG(INFO) << "benchmark ends at: " << jinq::common::Timestamp::now().to_format_str();
    LOG(INFO) << "cost time: " << cost_time << "s, fps: " << spec.loops / cost_time
              << ", mean: " << mean_ms << " ms, p50: " << p50_ms << " ms, p99: " << p99_ms << " ms";

    spec.handle_output(model_input, model_output, spec.image_path_of(argc, argv));
    return 0;
}

}  // namespace apps
}  // namespace jinq

#endif  // MORTRED_APPS_BENCHMARK_RUNNER_H
```

说明：
- `input_ok` 钩子用类型安全的 lambda 校验输入（`mat_input` 是非多态聚合体，不能对模板参数 `INPUT` 用 `dynamic_cast`，故校验逻辑由每个 spec 自带）。标准单图 spec 一律写 `spec.input_ok = [](const mat_input& in){ return !in.input_image.empty(); };`。
- 新增 warmup + mean/p50/p99 是评审中指出的方法学缺陷修复，属于**有意的输出增强**：总时长/fps 日志格式保持不变，新增三个统计字段。

- [ ] **步骤 2：迁移试点 `mobilenetv2_benchmark.cpp`**

全文替换为：

```cpp
// mobilenetv2 benckmark tool

#include <algorithm>

#include "apps/common/benchmark_runner.h"
#include "factory/classification_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::classification::std_classification_output;

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_classification_output> spec;
    spec.model_name = "mobilenetv2";
    spec.display_name = "mobilenetv2 classifier";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 1000;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(
            argc, argv, "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(
            argc, argv, "../demo_data/model_test_input/classification/ILSVRC2012_val_00000003.JPEG");
    };
    spec.make_model = [](const std::string& name) {
        return jinq::factory::classification::create_mobilenetv2_classifier<mat_input, std_classification_output>(name);
    };
    spec.handle_output = [](const mat_input&, const std_classification_output& out, const std::string&) {
        LOG(INFO) << "classify id: " << out.class_id;
        auto max_score = std::max_element(out.scores.begin(), out.scores.end());
        LOG(INFO) << "max classify socre: " << *max_score;
        LOG(INFO) << "max classify id: "
                  << static_cast<int>(std::distance(out.scores.begin(), max_score));
    };
    return jinq::apps::run_benchmark(argc, argv, spec);
}
```

- [ ] **步骤 3：构建 + 运行试点（行为对照）**

```bash
cmake --build build -j16
cd build && ./_bin/mobilenetv2_benchmark.out \
  ../conf/model/classification/mobilenetv2/mobilenetv2_config.toml
```

预期：输出与迁移前一致的 classify id/max score 日志，另多出 mean/p50/p99 行；fps 与迁移前同量级。

- [ ] **步骤 4：Commit**

```bash
git add src/apps/common/benchmark_runner.h src/apps/model_benchmark/classification/mobilenetv2_benchmark.cpp
git commit -m "refactor(benchmark): shared benchmark driver, migrate mobilenetv2"
```

---

### 任务 6：24 个标准 benchmark 迁移（按类别分 commit）

**文件：** benchmark 注册表前 8 行（分类×4、检测×7、分割×4、抠图×2、增强×3、深度×2、OCR×1、特征点×1）对应的 main cpp。

操作模式与任务 5 试点完全一致：每个 main 替换为"spec 声明 + `run_benchmark`"。各家族的 `handle_output` 直接把原文件循环之后的尾部代码搬进 lambda（原文件即数据源，删除前先搬）。典型形态：

```cpp
// 检测族（yolov5 为例；cls_nums 与后缀按原文件：yolo 系 80）
spec.handle_output = [](const mat_input& in,
                        const std_object_detection_output& out,
                        const std::string& image_path) {
    cv::Mat vis = in.input_image.clone();
    jinq::common::CvUtils::vis_object_detection(vis, out, 80);
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) +
                       "_yolov5_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/object_detection", output_file_name);
    cv::imwrite(output_path, vis);
    LOG(INFO) << "detection result image has been written into: " << output_path;
};
```

```cpp
// 分割族（bisenetv2 为例；hrnet/msocrnet/pphumanseg 同构，仅目录与后缀不同）
spec.handle_output = [](const mat_input&, const std_scene_segmentation_output& out,
                        const std::string& image_path) {
    cv::Mat color_seg_result;
    jinq::common::CvUtils::colorize_segmentation_mask(
        out.segmentation_result, color_seg_result, 80);
    std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
    output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) +
                       "_bisenetv2_result.png";
    std::string output_path = jinq::common::FilePathUtil::concat_path(
        "../demo_data/model_test_input/scene_segmentation", output_file_name);
    cv::imwrite(output_path, color_seg_result);
    LOG(INFO) << "segmentation result image has been written into: " << output_path;
};
```

循环数按注册表：densenet/resnet/mobilenetv2=1000、dinov2=100、pphumanseg=500、hrnet=10、深度×2=10、其余=100。默认图路径、输出子目录、后缀字符串逐文件从原 main 照搬（迁移 = 剪切粘贴原尾部，不是重写）。

增强族完整示例（enlightengan；matting/depth 族同构，仅字段名与保存目录不同）：

```cpp
// enlightengan benchmark tool

#include "apps/common/benchmark_runner.h"
#include "factory/enhancement_task.h"

using jinq::models::io_define::common_io::mat_input;
using jinq::models::io_define::enhancement::std_enhancement_output;

int main(int argc, char** argv) {
    jinq::apps::BenchmarkSpec<mat_input, std_enhancement_output> spec;
    spec.model_name = "enlightengan";
    spec.display_name = "enlightengan enhancementor";
    spec.usage = "exe config_file_path [test_image_path]";
    spec.loops = 100;
    spec.args_ok = jinq::apps::standard_args_ok;
    spec.input_ok = [](const mat_input& in) { return !in.input_image.empty(); };
    spec.make_input = [](int argc, char** argv, const toml::table&) {
        return jinq::apps::make_single_image_input(
            argc, argv, "../demo_data/model_test_input/enhancement/low_light/lol_test_1.png");
    };
    spec.image_path_of = [](int argc, char** argv) {
        return jinq::apps::standard_image_path(
            argc, argv, "../demo_data/model_test_input/enhancement/low_light/lol_test_1.png");
    };
    spec.make_model = [](const std::string& name) {
        return jinq::factory::enhancement::create_enlightengan_enhancementor<mat_input, std_enhancement_output>(name);
    };
    spec.handle_output = [](const mat_input&, const std_enhancement_output& out,
                            const std::string& image_path) {
        std::string output_file_name = jinq::common::FilePathUtil::get_file_name(image_path);
        output_file_name = output_file_name.substr(0, output_file_name.find_last_of('.')) +
                           "_enlightengan_result.png";
        std::string output_path = jinq::common::FilePathUtil::concat_path(
            "../demo_data/model_test_input/enhancement/low_light", output_file_name);
        cv::imwrite(output_path, out.enhancement_result);
        LOG(INFO) << "enhancement result image has been written into: " << output_path;
    };
    return jinq::apps::run_benchmark(argc, argv, spec);
}
```

提交切分：`refactor(benchmark): migrate <family> benchmarks to shared driver`，共 8 个 commit（分类/检测/分割/抠图/增强/深度/OCR/特征点）。每族迁移后运行一次代表性 benchmark（如 `yolov5_benchmark.out`、`bisenetv2_benchmark.out`）确认输出文件正常生成。

---

### 任务 7：lightglue + diffusion×4 折叠（钩子型）

**文件：** `feature_point/lightglue_benchmark.cpp`、`diffusion/{ddpm,ddim,cls_cond_ddim,ldm}_sampler_benchmark.cpp`

这两族不读标准单图 CLI，通过覆盖 `args_ok` / `make_input` 折叠：

- **lightglue**：`args_ok = [](int argc){ return argc==2||argc==4; }`（cfg + 可选两幅图）；`make_input` 搬原文件的双图读取与 `pair_mat_input` 组装（默认 `match_test_01/02.jpg`，两幅图都非空才返回有效输入）；`input_ok` 检查 `src_input_image` 与 `dst_input_image` 均非空；`handle_output` 搬原文件尾部保存逻辑。
- **diffusion×4**：`args_ok` 按各自原文件的参数个数（ddim 为 2-5 个参数）；`make_input` 搬原文件的 `sample_size/steps/save_all_mid_results` 解析与输入 struct 组装；`make_model` 用 `[](const std::string&){ return std::unique_ptr<...>(new DDIMSampler<INPUT, OUTPUT>()); }` 形式替代工厂（原文件即直接构造）；`handle_output` 搬原尾部的采样图保存；`input_ok = [](const std_ddim_input&){ return true; }`（输入无图像概念）。

验证：`cmake --build build -j16` 后逐个运行 4 个 diffusion benchmark 与 lightglue benchmark（权重/engine 本机已就绪），确认生成的结果图与迁移前一致。

Commit：`refactor(benchmark): fold lightglue and diffusion benchmarks into driver`

---

### 任务 8：文档、一致性校验与全量回归

**文件：**
- 修改：`docs/repository-layout.md`（Source tree 一节补 `apps/common/` 两行；确认 Server executables 表中每个 `src/apps/server/...` 路径仍存在——路径未动，应全部有效）
- 修改：`docs/how_to_add_new_server.md`（Step 2/3 改写为"写一个 spec 条目 + register_creator"，引用 `generic_ai_server.h` 与注册表实例）
- 修改：`README.md`（无 CLI 变化，仅需在 Benchmark 一节补一句 warmup/p50/p99 说明，可选）

- [ ] **步骤 1：文档更新**（如上）
- [ ] **步骤 2：一致性脚本**

```bash
python3 scripts/check_consistency.py
```

预期：无错误（该脚本校验 repository-layout 引用路径、conf/server→src/apps/server 目录映射、openapi 同步——本计划不动 exe 名与目录结构，全部应通过）。

- [ ] **步骤 3：全量回归**

```bash
cmake --build build -j16
ctest --test-dir build --output-on-failure          # 全部单测 + 契约 + golden
python3 scripts/check_consistency.py
bash scripts/check_repo_clean.sh
```

- [ ] **步骤 4：去重效果度量（计划目标的验收指标）**

```bash
find src test -name '*.h' -o -name '*.inl' -o -name '*.cpp' -o -name '*.cc' | xargs wc -l | tail -1
find src test \( -name '*.h' -o -name '*.inl' -o -name '*.cpp' -o -name '*.cc' \) | wc -l
```

基线：44,703 行 / 259 文件。验收目标：**净削减 ≥ 7,000 行**（server 侧约 -5,900：删 22 对类文件约 -5,870 行、factory 净增约 350；apps main 侧约 -1,700；benchmark 侧约 -1,800；新增两个公共头约 +450）。

- [ ] **步骤 5：Commit**

```bash
git add docs README.md
git commit -m "docs: registry-driven server/benchmark architecture"
```

---

## 风险与回滚

- **行为锚点**：HTTP 响应格式由 `response_schema_test` + `openapi_consistency_test` + golden JSON 保护；模型行为由 `model_golden_test` 保护；目录/产物契约由 `check_consistency.py` 保护。四道门全绿即行为等价的强证据。
- **提交粒度**：任务 3/6 按任务类别切 commit，任一类别异常可单独 revert。
- **已知有意变更**（均已在对应任务标注）：benchmark 新增 warmup 与 mean/p50/p99 统计；factory 注册名统一为 `"server"`（进程内注册表，无外部消费者）；`libserver.so` 变为纯头文件载体（无导出符号，链接关系不变）。
- **不迁移清单（防	scope 蔓延）**：clip/sam×3/bytetrack 五个多阶段 benchmark、`model_tools/trt_converter`、web_console——它们不属于同构重复。

## 自检（writing-plans 清单）

1. **规格覆盖度**：22 个 server（注册表 22 行 → 任务 2 迁移 1 行 + 任务 3 迁移 21 行，8 个类别提交全覆盖）；34 个 benchmark（29 条折叠 = 任务 5 试点 1 + 任务 6 标准 24 + 任务 7 钩子 5；5 条保留有明确理由与证据）；"注册表驱动"（AiServerSpec + register_creator + BenchmarkSpec）；"削减重复代码"（验收指标 ≥7,000 行，含度量命令）。✓
2. **占位符扫描**：任务 3/6/7 的"照搬原尾部"指向具体的现存文件与具体行段（循环之后的输出处理块），是机械剪切操作而非待定设计；所有新组件（register_creator、generic_ai_server、两个 main 驱动、两个家族 handler 示例）均给出完整代码。✓
3. **类型一致性**：`AiWorkerPtr<OUTPUT>` 与 22 个现有 `decltype(create_X<base64_input, OUTPUT>(""))` 同型；`AiResponseFiller` 与 `response_serializers.h` 的函数签名逐一核对；`BenchmarkSpec<INPUT, OUTPUT>` 与 `BaseAiModel<INPUT, OUTPUT>::run` 签名一致；`run_model_server_main` 的 toml 访问方式与现有 main 逐字一致。✓
