# P0 修复进度卡片（已完成）

> 单人顺序执行：0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8。
> P0-1（batch infer 优化）与 P0-4 第 3 步（sampler 迁移收敛）按计划**押后**到 batch 战役。
> 全部工作已合并 main（HEAD = 5d45c4ee）。

**状态图例**：✅ 完成并合并 ｜ 🚫 实测降级（不修，附证据）｜ ⏸ 押后

## 总览

| 阶段 | 内容 | 提交 | 状态 | 结论 |
|---|---|---|---|---|
| 0 | 基线采集 | 无 | ✅ | 数字见下表；为 4/5/6 三次降级提供了依据 |
| 1 | P0-5 draining 摘流 | f13f9c38 | ✅ | begin_drain → /ready 503 + 新请求 503 + 在途 200，20 轮 e2e |
| 2 | P0-3b 复活竞态 | be67c034 | ✅ | spawn 前复查 wanted/stopping；30 轮不变量守护 |
| 3 | P0-3a+c LogBuffer | 084f101b | ✅ | respawn 归档不截断；spawn 零等待；**附带修复 handle_exit 缺 notify 的潜伏 bug**（套件 180s→59s） |
| 4 | P0-2c MNN permute | 不修 | 🚫 | 三个候选全部 ≤1.1×；现有实现已被编译器优化；真实杠杆 = MNN 原生 NHWC（押后 batch 战役） |
| 5 | P0-2a metrics 锁 | 不修 | 🚫 | 154ns/请求当量、8 线程上限 1.02M/s——现实 QPS 下占比 <1%，非瓶颈 |
| 6 | P0-2b 热路径拷贝 | 40c59824 | ✅(缩减) | raw 路径 body move（全链路单拷贝）；insitu 手术实测 -13% < 30% 门，降级 |
| 7 | P0-4 diffusion 前两步 | ed317978 | ✅ | 采样状态局部化（并发隐患类清除）；进度条全删；golden NUMERICAL_PASSED |
| 8 | 收尾门禁 | 5d45c4ee | ✅ | -Werror 门禁清零（顺带修 3 处既有违例）；全量 check 95/98（3 个失败均与本轮无关，见下） |

## 阶段 0 基线（2026-10-21，main@f91b2644，RTX 2070/WSL2/Release/gpu）

| 指标 | 数值 |
|---|---|
| metrics 单线程 / 8 线程争用 | 30.8ns/call（154ns/请求当量）/ ~1.02M 请求当量每秒 |
| permute 1×640×640×3 f32 | v0=2.4ms；候选 v1/v2/v3 = 0.9×/1.0×/1.1× |
| HTTP 小图（33KB b64） | server total 2.5~3.5ms；client p50 3.4ms |
| HTTP 1.6MB b64 | total ~17ms；envelope 阶段 3.6~4.9ms |
| HTTP 8MB b64 | envelope 23~27ms（像素防护按设计拒绝） |
| envelope 拷贝链分解 | chunk 0.17 + parse 3.51 + images 0.20 + b64 1.18 = 5.07ms；insitu 候选 4.38ms |

收尾活体验证（合并后 main）：happy path 200 @6~19ms 无回归；SIGTERM 干净退出（"members destroying after drain"）。

## 门禁结果

- 全量 `check`（最终 HEAD=9772b067 重跑）：**97/98（98%）**。唯一失败 `p0_1_engine_fail_closed` AC3（TRT doctor 环境依赖，脚本型用例，与改动文件零交集，main 既有）。早前中段运行中的 gateway_e2e（/tmp 污染，已清）与 enlightengan segfault（本轮通过，确证环境 flake）均不再复现。
- `tests-only-werror`：**绿**（顺带修复 main 上 3 处既有违例：死函数 drop_env_entry、spawn_locked 失参、gpu_sampler name_len 残迹）。
- check_consistency / check_tsan_targets / check_version_single_source：全过。
- TSan：与 main 基线对比（42 vs 44 警告，同类既有误报；main 挂 2、分支挂 1）；本机 TSan 需 `setarch -R`（内核 6.18 ASLR 兼容性）。

## 移交清单（本轮发现、未修）

1. ~~契约管线两层漂移~~ **已修复（797e37e8）**：根因是 gen_openapi 的 `_ENVELOPE_DATA` 策略（msocrnet + dinov2 须用通用 EnvelopeData）从未在路由构建中实现——规则结构性不可通过；叠加 dump 自目录 id 迁移后从未再生（28→44 条）。现策略生效、dump/openapi 再生，`check_contract_sync` 全绿。
2. ~~目录 id 大小写不一致~~ **已修复（f42b6ba1）**：CLI 表名查找大小写不敏感（精确优先，命中变体打提示），host/port 解析同样容错；模型表边界保持精确匹配并给出清晰错误（该表名是全链路 catalog 键，深层容错会掩盖配置错误）。
3. `enlightengan_output_contract_unittest` segfault 与 `p0_1` AC3 doctor——main 既有，见上门禁结果。
4. 仓库里有一个**旧 stash**（`stash@{0}: On sub-agent-co-work: r4-wip-param-occupancy-async-before-designs`），非本轮产生，原样保留；本轮曾误弹导致半应用、已 `reset --hard` 恢复，stash 本体无损。
5. P1 清单（前次评审）：死代码清理、API key 加盐、非批路径队头阻塞、jinq→mortred 改名、mini_toml 静默吞错、gateway 重试/熔断等。

## P1 战役进行中（2026-10-22）

- P1-1 ✅ [kdf] salt（01372fe4）
- P1-2 ✅ 死代码三处（f1b8bfbc）
- P1-3 ✅ 调查结案，不改代码：非批路径"整请求持一个 worker、item 串行"是单 session 模型的固有约束——worker 是独占的模型实例，item 级交错物理不可能；worker_nums>1 时其他请求本就检出其他 worker 并行。真正的杠杆：worker_nums / max_request_items 配置、batch 战役的合批摊销。与阶段 4/5 同属"测量/分析后判定非代码缺陷"。
- P1-4 ✅ max_connections 默认 200→1024（LB 长连接 + 样例配置已是 500；保护模型的是 admission control 不是 socket 上限）
- P1-5 ✅ jinq→mortred 全库改名（294 文件含 .inl 与 C++ 生成器脚本；改名后全量 check 96/97，唯一失败为已知环境项 p0_1）
- P1-6 ✅ mini_toml 丢弃行 stderr 告警（466dba9d）
- P1-7 ✅ gateway 传输层重试 ×1 + breaker/180s 定性（afee5a7c）
- P1-8/9 ✅ 检测阈值单源 + status 8 英文文案（7d030fe1）
- 最终 HEAD 全量 check：**96/97（99%）**，唯一失败 p0_1_engine_fail_closed（TRT doctor 环境项，已定性）。rate_limiter 测试随死代码移除，用例总数 98→97。enlightengan 修复在全量套件稳定通过。

## 小项清理批次（2026-10-22，v1.1.0 发布后）

- ✅ v1.1.0 发布：版本单源 1.1.0、CHANGELOG 定版、tag v1.1.0 已推送（release 流水线自动构建镜像/tarball/SBOM）
- ✅ 测试盲区四个补齐（ff59defa）：mini_toml 4、sync_request_graph 2、gpu_sampler 4（parse_sample 转 public）、http_wire 5（纯函数 + 真实 Workflow task 的状态行/header）；mortredctl 延后有理由（薄 REST 客户端，服务端已全测）
- ✅ models 小项（7cf6dccf）：resnet 256 配方常量单源（三处联动）、DBNet 单输出回退、Tensor GPU 指针浅拷贝警告
- ✅ 卫生（449579bb）：23 处 Round-N/P1-x 账本注释改白话、models/catalog 并入 factory::catalog
- ✅ 记录性决策：max_queue 与 max_inflight 不合并（门控阶段不同，注释已写明）；environ token 可见性接受（同 uid 攻击者已有 ptrace/配置读取等更强原语）；factory create_xxx 样板保留（C++17 取函数指针惯用法）

## 下一战役（按用户计划）

batch infer 优化 + P0-4 第 3 步（4 个 sampler 收敛迁移到 BackendCvModel）+ 翻默认值的基准门禁（batch 实测反超 bs1 ≥10% 才翻）。

---

# 附：项目评估规则卡（供任意模型/人员按同一标准复评）

> 把本节整段复制给评估者即可。最近三次评估：2026-10-21 首评 8.0（HEAD 9b930bc6 前）、2026-10-21 P0 修复后 8.6、2026-10-22 P1+小项+v1.1.0 后 8.9（HEAD 7fe838d6 后）。

## 评估对象与立场

- 项目：`mortred_model_server`——C++17 AI 模型推理服务器（MNN / TensorRT / ONNX Runtime 三后端，Sogou Workflow 为 HTTP 引擎；supervisor 进程守护 + gateway 反向代理 + 每模型一进程）。
- 目标定位（评分锚点）：**GitHub 同类型个人项目的天花板，且能直接落地服务中型企业**；四个硬指标：高性能、高吞吐、好运维、易扩展。
- 评审立场：懒惰的资深工程师（过劳视角）——**过度设计与缺失基础保障同样扣分**；“测量/分析后判定非缺陷”的决策与代码修复同等计分，前提是有数字或证据记录。

## 硬性规则

1. **不读 `.git` 历史、不读 `docs/` 目录**；技术架构全靠代码梳理。
2. **代码与注释冲突时以代码为准**。
3. 每条扣分/加分必须给出 `file:line` 级证据；没有证据的评语不写。
4. 性能类指控必须先测量再定性：本机基线工具在 `scripts/microbench/`（run.sh 可复现历史数字）；实测不达改进门槛的“热点”一律判“非缺陷、不修”，不得凭直觉扣分或重写。
5. 门禁现状必须实测引用（不允许凭记忆）：全量 `cmake --build build --target check`；`cmake --build build/tests-only-werror`（-Werror 门）；`python3 scripts/check_contract_sync.py --dump-bin _bin/contract_dump`；`python3 scripts/check_consistency.py`；`python3 scripts/check_version_single_source.py`。TSan 需 `setarch $(uname -m) -R`（内核 ASLR 兼容）。

## 评分对象与权重

对 `src/` 下六个模块 + `test/` 各打 0-10 分，加权求和：

| 模块 | 权重 | 评估侧重 |
|---|---|---|
| `src/models`（推理核心，~15k 行） | 25% | 后端统一抽象、预处理/后置拷贝与数值、硬编码、GPU 资源与同步、新增一个模型家族的成本 |
| `src/server`（HTTP 服务层，~5k 行） | 20% | 请求全链路并发正确性、合批、背压/准入、优雅退出、热路径开销 |
| `src/control`（控制面，~9k 行） | 20% | 进程守护、限流、密钥安全、配置解析可见性、故障现场保留 |
| `test/`（~22k 行） | 15% | 断言真实性（非 smoke）、分层（单测/契约/e2e/压力/golden）、skip 语义、与 src 的覆盖对照、flaky 风险 |
| `src/common`（基础件） | 10% | 契约稳定性（wire 码/文案语言统一）、命名一致性、基础工具质量 |
| `src/factory`（目录/装配） | 5% | 目录单源化、扩展样板量 |
| `src/apps`（入口） | 5% | CLI 防御性、配置一致性校验 |

加权总分 = Σ(模块分 × 权重)，保留一位小数。

## 每个模块的输出格式

优缺点各至少 3 条（缺点必须含证据）；给出该模块分数与扣分明细（每项扣分标注“-x.x 及理由”）。全局输出：

1. 架构梳理（一段，靠代码）；
2. 模块评分表（含与上次评估的轨迹对比）；
3. 加权总分；
4. 未完成问题清单，分三类：**需要决策的 / 已排期无需决策的 / 押后到某战役的**。

## 当前已知的存留问题（评估时应核对而非重新发现）

- 唯一红用例：`p0_1_engine_fail_closed` AC3（TRT doctor 环境依赖，已定性）。
- diffusion 4 个 sampler 约 2000 行重复、未接入 BackendCvModel（押后到 batch infer 战役）。
- MNN 动态形状每请求 resizeSession 代价未测量；`Tensor::device_data` 浅拷贝约定（已加注释，无实际拷贝点）。
- 子进程 token 经 environ 对同 uid 可见（接受并记录）；429 限流 queue/inflight 两层保留（门控阶段不同，注释已写明）。
- mortredctl（cli_app.cpp）无直接单测（薄 REST 客户端，延后有理由）。
- stage_timing 走 /tmp 标志文件开关（opt-in，轻量保留）。

## 历史评估结论（供轨迹对比，禁止照抄分数）

- 首评 8.0：并发正确性 Research 级、吞吐出厂关闭、四处运维硬伤、jinq 命名空间、阈值硬编码。
- 二评 8.6：P0 五项闭环（draining/复活竞态/日志归档/notify 潜伏 bug/吞吐决策）；三个“热点”经实测降级为非缺陷（metrics 锁、nchw permute、in-situ 解析）。
- 三评 8.9：P1 九项闭环（kdf salt、死代码三处、队头阻塞定性、连接数默认、改名、mini_toml、gateway 重试、阈值单源、英文文案）+ v1.1.0 发布 + 测试盲区四个补齐 + 账本注释清零。

---

## 更新日志

- 2026-10-21 14:4x 卡片创建；阶段 0 启动。
- 2026-10-21 15:0x 阶段 0 完成；阶段 4/5 实测降级。
- 2026-10-21 15:3x 阶段 1 完成（f13f9c38）。
- 2026-10-21 15:5x 阶段 2 完成（be67c034）。
- 2026-10-21 16:1x 阶段 3 完成（084f101b，附带 notify 潜伏 bug）；阶段 6 完成（40c59824，insitu 降级）；阶段 7 完成（ed317978）。
- 2026-10-21 16:3x 阶段 8 完成：5 分支合并（000f232b→05fed40f）、全量门禁、WERROR 清零（5d45c4ee）、活体验证、移交清单落卡。
- 2026-10-21 17:0x 收尾：CHANGELOG [Unreleased] 条目（912d6c9f）；最终 HEAD 全量 check 97/98；合并分支清理；本地 main 领先 origin/main 15 个提交（未推送，等用户指示）。
- 2026-10-21 23:5x 用户决策 1-3 执行完毕：main 推送 origin（9b930bc6..eb246207）；契约管线修复（797e37e8）；CLI 大小写容错（f42b6ba1）。batch 战役、enlightengan/p0_1 追查、P1 排期、旧 stash 处置按用户指示暂缓。
- 2026-10-21 23:59x 用户决策批次二：enlightengan 间歇 segfault 根因修复（1fedfdea，preprocess 无 init 时 dtype 尾部解引用空 session，0/20→20/20）；旧 stash 已删；三件套微基准入库 scripts/microbench（run.sh 复现基线数字）。batch infer 按指示继续押后。
- 2026-10-22 P1 战役九项全部完成（01372fe4..7d030fe1）：加盐、死代码三处、队头阻塞定性、连接数默认、改名、mini_toml、gateway 重试、阈值单源、英文文案。单测/契约族全绿，改名后全量 check 96/97（唯一失败为 p0_1 环境项）。
- 2026-10-22 P1 完成验证：最终全量 check 96/97（99%）。
- 2026-10-22 新发现：**main 的 GitHub CI 红灯跨多个提交**（c22451a1 起 tests/gcc、container boot、cpu weights fetch、werror docker、inference paths 多 job 失败；部分对应本卡已修的 enlightengan/p0_1，但 werror docker 在本地全绿仍失败，需拉日志定性；CodeQL 亦有失败）。v1.1.0 的 release 流水线已正确触发（in progress）。最新 push 的 CI in progress。
- 2026-10-22 小项清理批次完成（7cf6dccf/449579bb/ff59defa），v1.1.0 已发布（3a15802e + tag）。
- 2026-10-22 小项批次验证：全量 check **100/101（99%）**（4 个新测试目标入列，总数 97→101；唯一失败仍为 p0_1 环境项）。
