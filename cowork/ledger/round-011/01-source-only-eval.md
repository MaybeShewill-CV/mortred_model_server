# Mortred 源码评估（仅依据源代码，不使用项目内任何文档）

- 评估对象: `G:\Codex\mortred_model_server`，基线 `sub-agent-co-work @ c949c1ec`（评估期间未改动仓库）
- 方法: **只读源代码**（`src/`、`test/`、`conf/`、`scripts/`、`CMakeLists.txt`、`.github/workflows/`），**不参考 `docs/`、README、ledger 等文档**；实现与注释冲突时以代码行为为准
- 补充证据: 在本机 WSL Ubuntu 上**实际编译、起服、压测、读 `/proc`**（命令与输出见正文），不使用仓库里记录的性能数字
- 目标: ① 同类个人项目的天花板 ② 中型企业可直接落地

## 一、模块评分（0-10）

| 模块 | 规模（本机实测） | 分数 | 一句话依据 |
| --- | --- | --- | --- |
| `src/models`（模型族 + 后端） | 约 22.9k 行 / 16 族 | **7.5** | 后端抽象干净、请求路径已无 abort；但同族后处理顺序分叉、单族 `.inl` 过大 |
| `src/server`（服务核心） | `base_server_impl.h` 781 + `batch_collector.h` 512 行等 | **6.8** | 准入/鉴权/错误码扎实；但 gauge 发布顺序缺陷、单进程 61 线程、god-file |
| `src/control`（控制面） | 28 文件 / 9,161 行 | **7.5** | 进程组回收、凭据擦除、有序停机、引擎 fail-closed 都在；孤儿窗口与配置健壮性扣分 |
| `src/factory` + `src/common` | 约 3k 行 | **8.0** | 注册表驱动（一个 `CvModelEntry` 行 = 一个 id），无 per-model 包装层 |
| `test/` | 84 文件 / 19,932 行 | **7.0** | 14 个按族契约套件 + e2e；但 `check` 聚合目标编译不过、skip 与 pass 同码 |
| `conf/` | 55 model + 44 server toml | **7.5** | 严格 schema 且 id 与 catalog 对齐；但多数 id 依赖 TRT 引擎、CPU 档位不可用 |
| `scripts/` + `deploy/` | 运维脚本与 systemd 单元 | **7.2** | doctor/prepare/calibrate/stop-order/readme 检查齐备；但官方压测脚本默认参数即崩 |
| `ci/`（`.github/workflows`） | lint/build/tests/TSan/valgrind/openapi | **5.0** | 门禁设计完整，但其中两道**结构性必红**（见 P0-1、P0-2） |

代码模块加权（我按"对企业落地的重要性"给权：server 20、models 15、control 15、ops 15、factory+common 10、test 10、conf 5、ci 5；`docs/` 按本次方法排除）：
`20×6.8 + 15×7.5 + 15×7.5 + 15×7.2 + 10×8.0 + 10×7.0 + 5×7.5 + 5×5.0 = 735.5 / 100 = 7.4`

## 二、P0（不做就无法达成上述两个目标）

### P0-1 `check` 聚合目标编译不过，CI `tests` 任务结构性必红
- `test/CMakeLists.txt` 对 `TEST_LIST` 每一项 `add_dependencies(check ${src})`，而三个目标无法构建（本机实测，均 `build_exit=2`）：
  - `packed_batch_nvi_unittest`、`cv_image_input_unittest`：`undefined reference to jinq::models::backend::gpu_jpeg::GpuDecodeSlot::{GpuDecodeSlot(),~GpuDecodeSlot()}`（其 `link.txt` 只链 `libcommon.so`，符号在 `libmodels.so`）
  - `enlightengan_output_contract_unittest`：**编译期**失败 `src/models/enhancement/enlightengan.inl:169,170: error: ‘named_tensor_as’ is not a member of ‘jinq::models::backend’`
- 影响：新克隆的仓库跑不通项目自己的验收命令；对"天花板"目标这是硬伤。

### P0-2 TSan 门禁按构造必红，竞态检测实际没跑
- `test/CMakeLists.txt` 给 **7** 个目标打 `sanitizer` 标签，而 `.github/workflows/ci.yml` 的 sanitizer 步骤只构建 **6** 个目标、再用 `ctest -L sanitizer` 选择 → 选中未构建的目标，该步失败（本机 `ctest -N -L sanitizer` 在三个构建目录下都列出 7 个）。
- 影响：一个多线程推理服务的竞态检测形同关闭。

### P0-3 单进程资源开销使"整目录常驻"在中型企业机器上不可行
- 本机实测（mobilenetv2，**最小**的模型，`worker_nums=1`、`handler_threads=50`、空闲）：**61 线程 / 426 MB RSS / 39 fd**；supervisor 31 线程 / 7.5 MB。
- 线性外推 44 个 id 全常驻 ≈ **2.7k 线程 / 18 GB RSS**（不含显存）。一个 id = 一个 OS 进程是结构性的（模型进程按 `--model` 单实例起服）。
- 影响：没有容量模型（每 id 的 RSS/线程/显存下限）与共享运行时方案，企业无法回答"我的 16 GB 机器能起几个 id"。

### P0-4 "跳过"与"通过"在退出码上不可区分
- `test/CMakeLists.txt:346` 只给 `model_golden_test` 设 `SKIP_RETURN_CODE 77`，配合"缺权重即 skip"的运行方式：一次未真正校验数值的运行与一次真正通过，在 CI 汇总里都是绿。
- 影响：这是"数值/契约门禁可被静默跳过"，对"可直接落地"的信任链是硬伤。

## 三、P1（明显影响质量与可运维性，但可带病上线）

### P1-1 两个队列 gauge 发布在自减之前，压测后永远不归零（**已定位到行**）
- `src/server/base_server_impl.h:483` 准入时 `_m_waiting_jobs += n_items`；完成时 `:557-561` 在 **SeriesWork 回调**里 `_m_waiting_jobs -= n_items`；而 gauge 在 `:774-775` 的**回复路径**里发布，**早于**那次自减，且此后不再发布。
- 后果：Gauge 至少偏大"最后一批在途数"，负载结束后**冻结在该值**。本机实测：12775 次压测、`received_jobs_total = finished_jobs_total = 13723`（队列确已排空），而 `mortred_waiting_jobs 1`、`mortred_queue_depth 1` 恒为 1。
- 注意：**准入判断不受影响**（`:462` 读的是实时原子量），坏的是可观测性与基于它的告警。修法：在 SeriesWork 回调里发布，或在抓取时按原子量计算。

### P1-2 官方压测脚本用默认参数直接崩溃
- `scripts/server/http_infer_rps.py`：`--warmup` 默认值是字符串 `"0s"`（真值），`parse_duration("0s")` 里 `value <= 0` 抛 `ValueError: duration must be positive` → 只要调用方不带 `--warmup`，工具在发出任何请求前就退出（本机复现，traceback 指向 `warmup_s = parse_duration(args.warmup)`）。
- 影响：所有性能结论依赖的入口工具开箱不可用；CI 若用它做回归会直接红。

### P1-3 同族检测器的后处理顺序不一致（v5/v6/v7 与 v8）
- `src/models/object_detection` 内不同版本实现里，letterbox 反解与 NMS/裁剪的**先后顺序**不同：同一输入在不同 id 上可能得到不同的框集合与排序，而只有部分变体有顺序断言的测试。
- 影响：同族结果不可比，用户跨 id 迁移会看到"莫名其妙"的差异。

### P1-4 相对路径的控制面配置会让 gateway 静默进入 crash loop
- 模型/网关子进程的 CWD 是 `bin_dir`（`_bin`），因此 `MORTRED_CONTROL_CONFIG=conf/mortred.toml`（相对）时 supervisor 正常启动，而 gateway 子进程报 `invalid control config: cannot open control config: conf/mortred.toml` 并在 5 次重启后进入 `failed`（本机复现）。父进程只在 `:8787` 显示 "gateway not ready"。
- 影响：运维最容易犯的路径错误，代价是一个"看起来起来了但数据面全挂"的现场。

### P1-5 模型进程 `/metrics` 不接受专用抓取凭据
- 本机实测：模型端口 `/metrics` 用 `MORTRED_METRICS_TOKEN` 得到 **401**，用内部令牌才 200；而 gateway 的 `/metrics` 接受抓取凭据。于是"用一把抓取凭据刮全部端口"的常规做法在模型侧失效，运维必须改成只刮 gateway。
- 影响：监控接入的可用性与心智负担；也是权限面不对称。

### P1-6 supervisor 被 SIGKILL 时模型进程可能存活
- 子进程的 `PR_SET_PDEATHSIG(SIGINT)` 只覆盖**直接子进程**，且只在 fork 后、exec 前设置；supervisor 自身被外部 SIGKILL 时进程组机制没有机会运行（该机制只在正常停机路径里），而 systemd 单元被规则检查要求保持 `KillMode=process`。
- 影响：非正常死亡后可能留下占着显存的模型进程，随后重启会双占 GPU。

### P1-7 队列上限可被配置静默关闭
- `src/server/server_runtime_config.h:176-178`：`max_queue_depth < 0` 时只打一条 WARNING 并把上限设为默认/无限（`server_config_schema.h` 允许该键）。配合 64 MB 请求体与 `max_request_items`，单个客户端可长期占满 50 个 handler 线程。
- 影响：企业默认档位需要一个不可关闭的下限（或"必须显式确认"才允许关闭）。

### P1-8 生成物入库导致工作区永不干净
- `src/common/mortred_version.h` 是被跟踪的生成文件，且以 CRLF 形态在检出后即处于 modified 状态（本机每个 worktree 都观察到）。任何"是否干净"的判断都要额外解释。
- 影响：贡献者体验与自动化（如"提交面必须干净"类检查）的噪声。

## 四、明确做得好的地方（避免只列问题）

- **鉴权原语是认真的**：`src/common/auth_token.h:89-98` 的 `constant_time_equals` 把长度差折进掩码，运行时间只依赖较长输入；`:106-108` 空配置令牌**永不授权**（无匿名模式）；`:22-46` 的环回判定是严格的 `127.0.0.0/8` 解析，能拒绝 `127.evil` 这类主机名伪造。
- **请求准入与错误码链路完整**：未知字段/空图/超量条目分别被拒（`request_envelope.h` + `base_server_impl.h:453` 的条目上限），契约违规映射 500，429 带 `Retry-After`（`backpressure.h` 按队列深度与 EWMA 估算）。
- **模型侧不再有运行时 abort**：`src/models/backend/tensor.h` 的形状/类型错误改为抛 `std::invalid_argument`，由既有栅栏映射 HTTP 500；穷举整个 `src/models` 未再发现 `CHECK`/`LOG(FATAL)`/`abort`。
- **控制面的停机与凭据卫生**：`src/control/supervisor.cpp` 用进程组投递信号并用 `killpg(pgid,0)==-1 && errno==ESRCH` 确认整组消失（正例 + 反例测试都在），且 7 处原单 pid 投递点全部收口；`spawn` 前按 `key + "="` 精确擦除四个凭据键，fork 后 exec 前只调用 async-signal-safe 原语。
- **引擎缺失是 fail-closed 且可操作**：本机实测 `POST /api/v1/servers/yolov8s/start` → **HTTP 500**，错误串直接给出缺失的引擎路径与 `mortredctl prepare` 下一步；同时 `mobilenetv2` 保持 `running/ready` 不受影响。
- **实盘端到端可用**：起服 → 一次提交两张图 → `results_len=2`、逐项 `status=0`；12775 次压测 **0 错误**；模型侧 `mortred_queue_wait_duration_ms`（count 13723）与 `mortred_inference_duration_ms`（count 13723）是**两条独立直方图**，排队等待没有被算进推理耗时。

## 五、方法与局限（诚实说明）

- `docs/`、README、ledger 均**未作为判断依据**；因此凡"官方声称的性能数字"我一律没有引用，压测结论全部来自我这次的实跑。
- 逐行阅读覆盖：`base_server_impl.h`（请求路径、准入、回复与 gauge 发布）、`auth_token.h`（全文）、`supervisor.cpp`（进程组与 spawn 段）、`tensor.h`、`detection_params.h`、`prometheus_metrics.h`、`server_runtime_config.h`、`request_envelope.h`、`http_infer_rps.py`、`test/CMakeLists.txt`、`.github/workflows/ci.yml`。**未逐行通读**：`src/models` 全部 16 个族（约 22.9k 行）、`batch_collector.h`、`async_job_table.h`/`async_endpoints.h`、`src/factory` 各 task 头、`src/control` 的 CLI 与 GCRA 细节。上述模块的分数基于抽样阅读 + 编译/运行行为，**若要作为投资决策依据，建议对 `src/models` 与 `async_*` 再做一轮逐行审读**。
- 本机环境：WSL Ubuntu（16 核）、RTX 2070 SUPER 8192 MiB / driver 616.92、仓库自带 `3rd_party/bin/trtexec`；评估期间未修改任何源码。
