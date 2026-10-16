# 模块：src-factory-common

- 路径: `src/factory`、`src/common`
- 权重: 10
- 分数: 8.0
- 基线: sub-agent-co-work @ `8993adb3`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **`src/factory`（17 文件、约 1.4k 行）就是"一个二进制服务 44 个 id"的机制所在**：12 个任务族各暴露一个 `catalog()` 常量表（classification、clip、diffusion、enhancement、feature_embedding、feature_point、matting、mono_depth、obj_detection、ocr、sam、scene_segmentation），`src/factory/model_catalog.h`（52 行）的 `create_model(catalog(), model_section)` 按条目建模型——例如 `src/factory/clip_task.h:31` 的 `catalog()` 与 `:39` 的 `create_model(...)`；另有 `cv_catalog.h`（68）、`base_factory.h`（87）与 `diffusion_model_adapter.h`（131）处理扩散族的适配。
- **`src/common`（16 文件、1729 行）职责单一、无明显越界**：`cv_utils.h`（454）、`response_envelope.h`（219）、`request_envelope.h`（207）、`base64.h`（141）、`stage_timing.h`（138）、`auth_token.h`（116）、`process_stop.h`（111）、`time_stamp.h/.cpp`、`status_code.h`（74）、`listen_policy.h`（58）、`file_path_util.h`、`request_size_limit.h`（26）、`mortred_version.h`（4）。
- **EX2 的机制已活体验证**（编排方实跑）：`./_bin/mortred-model-server.out --list` **退出码 0**，输出带表头的表格（列：`id` / `family` / `surface` / `display_name`），共 **44** 个 id，横跨 diffusion、segment_anything、enhancement、scene_segmentation、face_detection、ocr、classification、mono_depth_estimation 等族；网关侧路由是**按 catalog id 泛化**的（`src/control/gateway/gateway_app.cpp:288` `static const std::string kPrefix = "/v1/models/";`，`:297` 注释"Exact catalog id"），在 `src/control/` 内 grep 具体 id（`yolov8n|yolov8s|libface|mobilenetv2|sam_amg`）**只命中一处注释**（`src/control/catalog.h:41` 的举例），即加一个 id 不需改路由代码。
- 路径卫生：`src/factory` 与 `src/common` 下无 `__pycache__`、`.o`、`.orig`、`.bak` 等残留（实跑 `find` 确认）。
- 本模块的两个头文件是本轮多个条目的契约来源并被独立验证：`request_envelope.h:97`（未知字段拒绝文案，ST1 证据）、`request_size_limit.h`（第 9 轮 P0-2 的 body 上限）、`listen_policy.h`/`auth_token.h`（第 9 轮 gateway fail-closed 与 OP4 的基础）。

## 问题

1. **P2（本轮新发现，尚未入列）`src/common/mortred_version.h` 是入库的生成物，工作区恒为 CRLF-dirty**：该文件被 git 跟踪（4 行），但在 checkout 后即与索引不一致——表现为每个 worktree 的 `git status` 都有一条 ` M src/common/mortred_version.h`（本轮 P1-1/P1-2/P1-4/P1-5 四个实现者都必须在提交前手工排除它，P1-1 的 RD 还为此专门做了 `sed -i 's/\r$//'` 归一与 `--amend`）。出处：`git status` 于任意 worktree；`.gitattributes` 的 `* text=auto`；`git check-ignore` 显示它未被忽略。现在的行为：每个在该仓库工作的人都要额外记住"不要提交它"，而它一旦被误提交就会把构建号写进历史。一次外部验收即可确认的变化：`git clone` 后无论是否构建，`git status --porcelain` 都不再出现该文件的修改（做法二选一：把它改为构建期生成并加入 `.gitignore`，或在 `.gitattributes` 里对它显式声明 `-text`/`eol=lf` 使其与索引一致）。
2. 本轮锁定的 7 个 P1 条目**无一指向本模块的路径**（P1-1/P1-4 在 `src/models`、P1-2/P1-3 在 `src/control`、P1-5/P1-6/P1-7 在 `src/server`/`test`/`conf`/`docs`），故除上一条 P2 外，本模块**没有未关闭的入列项**。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| EX2（在已有家族的 `catalog()` 加 id 后 `--list` 出现该 id，且 gateway 不需改路由） | **已满足** | 活体：`./_bin/mortred-model-server.out --list` 退出 0、44 个 id、含 `family` 列；`gateway_app.cpp:288` 的 `/v1/models/` 前缀路由 + `src/control/` 内无硬编码 id（仅 `catalog.h:41` 一处注释举例） |
| EX1（文档不得要求每个模型单独做服务器二进制；步骤与 `catalog()`、`--list` 一致） | **待编排方判定（归属 `docs` 模块）** | 本记录只提供机制侧证据：`catalog()` + `--list` 确实构成"加 id 不加二进制"的路径；文档是否这么写应由 `docs` 模块记录判定，本记录不代替 |
| AV1（`type=mnn`/`type=onnx` 的已服务 id 能单独拉起并完成一次合法推理） | **待编排方判定（归属 `src-models`/`test`）** | 本机缺 MNN/ONNX 权重时相关用例 SKIPPED（P1-4 QA 已记录 `backend_unittest` 5 例跳过）；不属本模块路径 |
| ST1（未知字段/空图/错误 token 不得得到成功） | **已满足（本模块只承担其中一半）** | 未知字段与信封校验的实现在 `src/common/request_envelope.h:97`；完整判定见 `07-score-check.md` 的 ST1 行 |

## 评分依据

- 给 8.0 而不是更高：① 上表第 1 条是一个**真实且每次都在干扰开发者**的入库生成物问题（本轮四个实现者都被它绊到，属"每次提交都要手工排除"的持续性摩擦），按 rubric 的 10 衡量，一个小组照文档操作时不该需要这种隐性知识；② `src/common/cv_utils.h` 454 行是一个杂项集合（编解码、几何、颜色空间混装），是本模块唯一明显"该拆"的文件；③ `src/common` 只有一个 `time_stamp.cpp`，其余全是 header-only inline 实现，缺少编译期隔离，改一个头会波及大量 TU。
- 给 8.0 而不是更低：③ 的机制层面**干净且是本项目的核心价值**——`catalog()` + `create_model()` + `--list` 让一个二进制服务 44 个 id，并且这条路径已有活体证据与"网关零硬编码 id"的交叉验证；④ 两个路径无残留垃圾文件；⑤ 本模块的契约头文件（`request_envelope.h`、`request_size_limit.h`、`listen_policy.h`、`auth_token.h`）正是本轮 ST1/ST2/OP4 的证据来源，说明它们被真实使用而非摆设。
- 参照系：清掉 `mortred_version.h` 的入库生成物问题（一次 `.gitignore`/`.gitattributes` 改动）后，本模块有理由到 8.5。
