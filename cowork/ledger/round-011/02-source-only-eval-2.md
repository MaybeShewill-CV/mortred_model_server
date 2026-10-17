# Mortred 源码评估 · 第二次（仅依据源代码，不使用项目内文档）

- 基线: `sub-agent-co-work @ 3c9f49a3`（距上次评估 `80a394bb` 之间落入了 round-13 与 round-14 两批提交）
- 方法: 与上次完全一致——只读源代码 + 本机自建/自跑取证（编译、起服、压测、读 `/proc`、跑项目自带检查脚本）；**不把 `docs/`、README、账本或提交信息当成结论依据**
- 本次重点: 逐条复核上次给出的 4 个 P0 + 8 个 P1，并检查两批新提交是否引入新问题

## 0. 先更正我上次的算术错误

上次我写"加权和 735.5 → 7.4 分"。重新逐项相加：`20×6.8 + 15×7.5 + 15×7.5 + 15×7.2 + 10×8.0 + 10×7.0 + 5×7.5 + 5×5.0 = 681.5`，**应为 6.8 分**，不是 7.4。所以本次的对比基准是 **6.8**，而不是 7.4。（权重与分项不变，纯属求和错误。）

## 1. 上次 12 条问题的复核结果

| 编号 | 上次判定 | 本次实测 | 证据（本次亲手执行） |
| --- | --- | --- | --- |
| P0-1 三个测试目标拖垮 `check` | 红 | **两个已修好；第三个未实证** | `packed_batch_nvi_unittest`、`cv_image_input_unittest` 现在用 `EXTRA_SRCS` 直接编译 `gpu_jpeg_decoder.cpp`（`test/CMakeLists.txt:176-178,193-194`），我实际构建二者均 **`build_exit=0`**；`enlightengan` 缺的 `named_tensor_as` 已在 `src/models/backend/model_runtime.h:102` 实现，该目标仅在 `MORTRED_BUILD_FULL` 下注册（`:127-131`）且有 `EXTRA_SRCS`（`:236`）——但我**没有跑 FULL 全量构建**，故不作实证 |
| P0-2 TSan 门按构造必红 | 红 | **已修好，且加了自检** | label 条件（`test/CMakeLists.txt:361-367`）与 CI 构建列表（`ci.yml:330`）同为 7 个目标；`python3 scripts/check_tsan_targets.py` → **退出 0**，输出 `TSan target lists agree across labels / preset / CI` |
| P0-3 单进程开销导致整目录常驻不可行 | 不可行 | **半修：线程解决，内存未解决** | `handler_threads` 默认 50→**8**（`server_runtime_config.h:47`），出厂 conf=16；实测同一最小模型：线程 **61 → 27**，RSS **426 MB → 418 MB（基本不变）**；新增 `src/control/occupancy_gate.h:531-545` 的 RSS/RAM 门（`ram_reserve_pct=15`，自+兄弟 RSS 与主机内存比较）。44 id 线性外推仍 ≈18 GB |
| P0-4 skip 与 pass 同码 | 同码 | **半修** | `parse_duration` 现只拒绝 `<0`；`model_golden_test` 只在"全部缺数值"时返回 77（`test/model_golden_test.cc:715,722,733`）；CI 的 cpu-profile 作业注释明确 `Skip is fail`（`ci.yml:368`）并在 golden 作业用 `--allow-skips` 显式放行（`:972`）。但 ctest 语义下 77 仍渲染为 `Skipped`（可见、非失败） |
| P1-1 队列 gauge 永不归零 | 恒为 1 | **已修好（实测）** | `base_server_impl.h:492-494` 准入即发布、`:571-573` 自减后**再发布**；实测 3558 请求压测后 `received_jobs_total = finished_jobs_total = 4701`、`mortred_waiting_jobs 0`、`mortred_queue_depth 0` |
| P1-2 官方压测脚本默认参数即崩 | 崩 | **已修好** | `parse_duration` 的判据从 `value <= 0` 改为 `value < 0`；本轮多次以 `--warmup 2s` 与默认路径运行均 `exit 0` |
| P1-3 同族后处理顺序分叉 | 分叉 | **已修好** | `src/models/object_detection/yolov5_detector.inl:89` 先 `unmap_letterbox_detections`，`:91` 再 `finalize_detections`（含 NMS）；round-14 又把 detection/matting/seg 的尾部统一 |
| P1-4 相对控制面配置让 gateway 静默崩溃 | 存在 | **仍然存在** | `src/control/supervisor/supervisor_app.cpp:930` 仅 `lexically_normal()`（**不转绝对**），`control_config.cpp:120` 仍直接报 `cannot open control config: <path>`；子进程 CWD 是 `_bin` |
| P1-5 模型 `/metrics` 不接受抓取凭据 | 401 | **仍然存在，且原因变成"两处修复互相抵消"** | 代码侧 `base_server_impl.h:342` 已读 `MORTRED_METRICS_TOKEN`，但 round-14 的**子进程环境 allowlist**（`supervisor.cpp:441-448`）只放行 PATH/HOME/LD_*/CUDA_*，**不含该键**，模型侧因此拿不到它 → 实测 `MORTRED_METRICS_TOKEN → 401`、`MORTRED_INTERNAL_TOKEN → 200`、`MORTRED_API_TOKEN → 401` |
| P1-6 supervisor 被 SIGKILL 留下孤儿 | 有窗口 | **改了但引入新矛盾** | `deploy/mortred-supervisor.service:46` 已是 `KillMode=control-group`；而 `scripts/stop_order_record.py:141,143` 要求"恰一行 `KillMode=process`"，并把 `control-group` 列为**必须失败**的负例（`:168`）→ 我实跑 `--check` 得 **`check_exit=1` / "stop order check did not match"**：项目自带门禁对出厂单元文件报失败 |
| P1-7 队列上限可被配置静默关闭 | 静默关闭 | **已修好** | `server_runtime_config.h:176-179`：`max_queue_depth <= 0` 现在夹到正默认值并打 WARNING（默认 32，`:54`） |
| P1-8 生成物入库 | dirty | **仍然存在** | `src/common/mortred_version.h` 仍被 git 跟踪，`CMakeLists.txt:16-17` 仍 `configure_file` 写回源码树 |

**小结: 12 条中 8 条已修（P0-1 部分、P0-2、P1-1、P1-2、P1-3、P1-7 + P0-3/P0-4 半修），2 条仍存在（P1-4、P1-8），1 条改出新矛盾（P1-6），1 条被自己的新改动抵消（P1-5）。**

## 2. 本次新发现（两批新提交带来）

### N-A（P1）构建输出目录被所有构建目录共享，切换 profile 会静默替换二进制且不再重链
- `CMakeLists.txt:32-35`：`MORTRED_BIN_OUTPUT_DIR` 默认 `${PROJECT_ROOT_DIR}/_bin`、`MORTRED_LIB_OUTPUT_DIR` 默认 `${PROJECT_ROOT_DIR}/_lib`，并直接赋给 `CMAKE_RUNTIME_OUTPUT_DIRECTORY` / `CMAKE_LIBRARY_OUTPUT_DIRECTORY`。于是**任何**构建目录（`build`=gpu/FULL、`build-eval`=cpu/tests-only）都写同一个 `_bin/_lib`。
- 我实测到的两次后果：① 用 `-DMORTRED_BUILD_PROFILE=cpu` 配置一个新目录并构建测试目标后，`_bin/mortred-model-server.out` 变成 **cpu profile**，运行期报 `create tensorrt session for [MOBILENETV2] failed: tensorrt backend is not compiled into this build (cpu profile)`，所有 TRT id 无法服务；② 随后对 gpu 目录执行 `cmake --build build --target mortred-model-server.out` **退出 0 却没有 `Linking` 行**——因为共享输出路径上那个（更晚的）cpu 二进制使目标被判定为最新，**正确 profile 的构建被静默跳过**。只有删掉 `_bin/mortred-model-server.out` 才真正重链（我这样做了，才有后面的一手数据）。
- 影响：本地/自托管复现性。CI 每个作业是干净 runner 所以看不见，但对"按文档在开发机切换 profile"的人是个陷阱。

### N-B（P1 的一部分）见 P1-6：门禁脚本与交付物的期望相反
`stop_order_record.py` 把 `KillMode=control-group` 当成"应当失败"的样例，而 round-13 为了修 SIGKILL 孤儿问题正是把它改成了 `control-group`。两者必须有一方更新（我的判断：检查器应接受 `control-group`，因为它只在 systemd 硬杀时兜底，正常停机仍由 supervisor 两阶段完成）。

## 3. 模块评分（同一组权重与方法）

权重沿用上次（server 20、models 15、control 15、ops 15、factory+common 10、test 10、conf 5、ci 5；`docs` 按"不读文档"原则不评）。

| 模块 | 上次 | 本次 | 变化依据 |
| --- | --- | --- | --- |
| `src/server` | 6.8 | **7.8** | gauge 发布顺序修正（实测归零）、模型 `/metrics` 代码支持抓取凭据、`max_queue_depth<=0` 夹紧、`handler_threads` 默认 50→8；仍扣：787 行 god-file + 512 行 collector、allowlist 把凭据截断导致上一行修复失效、async 保留策略未审 |
| `src/models` | 7.5 | **8.0** | 同族尾部统一（unmap→NMS 已实证）、TRT `ensure` 失败置 abandon、非 JPEG/PNG 预解码体积门 |
| `src/control` | 7.5 | **7.8** | 子进程 env 改 **allowlist**（比 drop-list 正确）、`KillMode=control-group`、RSS/RAM 占用门；扣：allowlist 漏 `MORTRED_METRICS_TOKEN`、相对配置未绝对化、与 stop-order 检查器矛盾 |
| `ops` | 7.2 | **7.5** | 压测脚本默认参数修复、容量表与 RSS 地板；扣：共享 `_bin` 复现性陷阱 |
| `factory`+`common` | 8.0 | **7.8** | 变化不大；`mortred_version.h` 仍是生成物入库 |
| `test` | 7.0 | **8.0** | `check` 的两个不可链接目标已实证可构建、TSan 标签与 CI 单一真源且有守卫脚本、golden skip 语义收紧 |
| `conf` | 7.5 | **8.0** | `handler_threads` 16、`max_queue_depth` 正默认、pack 与容量字段更新 |
| `ci` | 5.0 | **7.8** | 两道结构性红门都已修并加自检脚本，另有 cpu-profile 托管 golden 作业；扣：我无法在本地实跑 GitHub 作业，未验证整体绿 |

加权和：`20×7.8 + 15×8.0 + 15×7.8 + 15×7.5 + 10×7.8 + 10×8.0 + 5×8.0 + 5×7.8 = 156 + 120 + 117 + 112.5 + 78 + 80 + 40 + 39 = 742.5`

**本次综合分 = 7.4（742.5/100 = 7.425 → 7.4）**，相对更正后的上次基准 **6.8 → 7.4，+0.6**。

## 4. 现在的 P0 / P1（按对企业落地与"同类天花板"的阻断程度排序）

### P0（仍阻断目标）
1. **单进程内存地板 418 MB（最小模型、`worker_nums=1`）**：线程已从 61 降到 27，但 RSS 几乎没动。44 个 id 全常驻仍需 ≈18 GB。新增的 RAM 门只是**拒绝**超预算启动，并没有降低单进程成本。要么给出"按容量起子集"的官方部署档位（已有一张容量表，但缺"推荐常驻子集"），要么做共享运行时。**这是能否进中型企业的分水岭。**
2. **FULL 档位下 `check` 是否真能跑完尚未被证明**：两个 jpeg 目标已实证修好、enlightengan 的依赖符号已补齐，但我没跑 FULL 全量；在这一点被实证前，"门禁变绿"只能算 2/3 成立。

### P1（按影响排序）
3. **`MORTRED_METRICS_TOKEN` 在 supervisor 托管的模型进程里不可用**（`supervisor.cpp:441-448` allowlist 漏键，实测 401/200）——r13 的修复被 r14 抵消，属于回归。
4. **`stop_order_record.py --check` 对出厂 unit 返回 1**（`KillMode=control-group` vs 检查器要求 `process`）——项目自带门禁与交付物互相矛盾。
5. **共享 `_bin/_lib` 使 profile 切换静默替换二进制、且正确构建会被判定为最新**（`CMakeLists.txt:32-35`，实测两次踩中）。
6. **相对路径 `MORTRED_CONTROL_CONFIG` 让 gateway 静默 crash-loop**（`supervisor_app.cpp:930` 仅 `lexically_normal`）。
7. **`src/common/mortred_version.h` 生成物入库**，检出即 dirty（`CMakeLists.txt:16-17`）。
8. **ctest 语义下 skip 仍非失败**（77 渲染为 Skipped）：CI 有一个作业把它当失败，但本地 `check` 聚合仍可能"全绿带跳过"。

## 5. 依然值得肯定的地方（本轮复核仍成立）

- 鉴权原语（`auth_token.h` 常数时间比较、空配置不授权、严格 127/8 解析）与准入链路（条目上限、`Retry-After`、契约违规映射）依旧扎实。
- 模型侧无运行时 abort；进程组回收与整组确认（正例+反例）、凭据按 allowlist 白名单传递（方向正确）、TRT `ensure` 失败置 abandon。
- 端到端可用性提升明显：本轮 3558/7383 请求 **0 错误**，`queue_wait` 与 `inference` 两条直方图独立且都被填充，gauge 能正确归零。
- 修复带**守卫**：TSan 目标列表有 `scripts/check_tsan_targets.py` 做单一真源校验——这比"改对一次"更值钱。

## 6. 局限（必须说明）

- **未逐行通读**：`src/models` 全部 16 族、`batch_collector.h`、`async_job_table.h`/`async_endpoints.h`、`src/factory` 各 task 头、`src/control` 的 CLI 与 GCRA 细节；这些模块的分数来自抽样阅读 + 编译/运行行为。
- **未实跑 GitHub Actions**：`ci` 的 7.8 分基于构建列表一致性与守卫脚本，不等于"我已看到全绿"。
- **未跑 FULL 全量构建**（enlightengan 目标）。
- 本轮我在验证过程中曾误读一个后台作业的退出码（用的目标名 `mortred-model-server` 不存在，实际目标名带 `.out` 后缀），并因此短暂删掉了 `_bin/mortred-model-server.out`；已用正确目标名重建并确认 `Linking CXX executable` 与 mtime，随后所有实测都在 gpu 版二进制上完成。此事本身也构成了 N-A 的证据链。
