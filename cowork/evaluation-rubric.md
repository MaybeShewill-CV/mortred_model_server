# mortred_model_server 评估规则卡

> 本文件自包含：把整份内容复制给任意模型/人员，即可按同一标准复评本项目。
> 最近三次评估轨迹：2026-10-21 首评 **8.0** → 2026-10-21 P0 修复后 **8.6** → 2026-10-22 P1+小项+v1.1.0 后 **8.9**。

## 评估对象与立场

- 项目：`mortred_model_server`——C++17 AI 模型推理服务器（MNN / TensorRT / ONNX Runtime 三后端，Sogou Workflow 为 HTTP 引擎；supervisor 进程守护 + gateway 反向代理 + 每模型一进程）。
- 目标定位（评分锚点）：**GitHub 同类型个人项目的天花板，且能直接落地服务中型企业**；四个硬指标：高性能、高吞吐、好运维、易扩展。
- 评审立场：懒惰的资深工程师（过劳视角）——**过度设计与缺失基础保障同样扣分**；“测量/分析后判定非缺陷”的决策与代码修复同等计分，前提是有数字或证据记录。

## 硬性规则

1. **不读 `.git` 历史、不读 `docs/` 目录**；技术架构全靠代码梳理。
2. **代码与注释冲突时以代码为准**。
3. 每条扣分/加分必须给出 `file:line` 级证据；没有证据的评语不写。
4. 性能类指控必须先测量再定性：基线工具在 `scripts/microbench/`（`run.sh` 可复现历史数字）；实测不达改进门槛的“热点”一律判“非缺陷、不修”，不得凭直觉扣分或重写。
5. 门禁现状必须实测引用（不允许凭记忆）：
   - 全量测试：`cmake --build build --target check`
   - -Werror 门：`cmake --build build/tests-only-werror`
   - 契约链：`python3 scripts/check_contract_sync.py --dump-bin _bin/contract_dump`
   - 一致性：`python3 scripts/check_consistency.py`
   - 版本单源：`python3 scripts/check_version_single_source.py`
   - TSan 需 `setarch $(uname -m) -R`（内核 ASLR 兼容）

## 评分对象与权重

对 `src/` 下六个模块 + `test/` 各打 0-10 分，加权求和（总分保留一位小数）：

| 模块 | 权重 | 评估侧重 |
|---|---|---|
| `src/models`（推理核心，~15k 行） | 25% | 后端统一抽象、预处理/后置拷贝与数值、硬编码、GPU 资源与同步、新增一个模型家族的成本 |
| `src/server`（HTTP 服务层，~5k 行） | 20% | 请求全链路并发正确性、合批、背压/准入、优雅退出、热路径开销 |
| `src/control`（控制面，~9k 行） | 20% | 进程守护、限流、密钥安全、配置解析可见性、故障现场保留 |
| `test/`（~22k 行） | 15% | 断言真实性（非 smoke）、分层（单测/契约/e2e/压力/golden）、skip 语义、与 src 的覆盖对照、flaky 风险 |
| `src/common`（基础件） | 10% | 契约稳定性（wire 码/文案语言统一）、命名一致性、基础工具质量 |
| `src/factory`（目录/装配） | 5% | 目录单源化、扩展样板量 |
| `src/apps`（入口） | 5% | CLI 防御性、配置一致性校验 |

## 每个模块的输出格式

- 优缺点各至少 3 条（缺点必须含证据）；
- 该模块分数与扣分明细（每项扣分标注 “-x.x 及理由”）。

全局输出四段：

1. 架构梳理（一段，靠代码）；
2. 模块评分表（含与上次评估的轨迹对比）；
3. 加权总分；
4. 未完成问题清单，分三类：**需要决策的 / 已排期无需决策的 / 押后到某战役的**。

## 当前已知的存留问题（评估时应核对而非重新发现）

- 唯一红用例：`p0_1_engine_fail_closed` AC3（TRT doctor 环境依赖，已定性）
- diffusion 4 个 sampler 约 2000 行重复、未接入 BackendCvModel（押后到 batch infer 战役）
- MNN 动态形状每请求 resizeSession 代价未测量；`Tensor::device_data` 浅拷贝约定（已加注释，无实际拷贝点）
- 子进程 token 经 environ 对同 uid 可见（接受并记录）；429 限流 queue/inflight 两层保留（门控阶段不同，注释已写明）
- mortredctl（cli_app.cpp）无直接单测（薄 REST 客户端，延后有理由）
- stage_timing 走 /tmp 标志文件开关（opt-in，轻量保留）

## 历史评估结论（供轨迹对比，禁止照抄分数）

- 首评 8.0：并发正确性 Research 级、吞吐出厂关闭、四处运维硬伤、jinq 命名空间、阈值硬编码
- 二评 8.6：P0 五项闭环（draining / 复活竞态 / 日志归档 / notify 潜伏 bug / 吞吐决策）；三个“热点”经实测降级为非缺陷（metrics 锁、nchw permute、in-situ 解析）
- 三评 8.9：P1 九项闭环（kdf salt、死代码三处、队头阻塞定性、连接数默认、改名、mini_toml、gateway 重试、阈值单源、英文文案）+ v1.1.0 发布 + 测试盲区四个补齐 + 账本注释清零
