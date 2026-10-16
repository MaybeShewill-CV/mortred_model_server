# 模块：ops

- 路径: `scripts/`、`deploy/`、`mortredctl`
- 权重: 15
- 分数: 8.0
- 基线: sub-agent-co-work @ `aab443c5`
- 记录人: 编排方（单一执行方模式；本份由编排方直接执行，因并行子代理连续失败，非独立 `cv-cto` 子代理）

## 已具备

- **`scripts/` 共 70+ 个可执行件**，覆盖运维全链：引擎与打包（`build_trt_engine.sh`、`convert_trt_engines.sh`、`trt_engine_builder.cc`、`trt_engine_inspect.cc`、`calibrate_pack.py`、`pack_trt.py`、`pack_occupancy.py`、`product_static_bs1.py`、`gen_trt_batch_profiles.py`、`patch_onnx_dynamic_batch.py`）、依赖与镜像（`install_deps.sh`、`install_host_common.sh`、`setup_full_deps.sh`、`bootstrap.sh`、`refresh_base_image_digests.sh`、`docker_entrypoint.sh`）、发布（`make_release_tarball.sh`、`release_dry_run.sh`、`tarball_install.sh`）、校验（`check_consistency.py`、`check_version_single_source.py`、`check_contract_sync.py`、`check_base_image_digests.py`、`check_repo_clean.sh`、`verify_deployment.sh`、`verify_gpu_gate_evidence.py`、`verify_libface_yunet.py`、`golden_drift_check.py`）、记录（`stop_order_record.py`、`resident_pair_record.py`、`model_dx_metrics.py`、`coverage_http_report.py`）、权重（`fetch_weights.py`、`gen_weights_manifest.py`）。
- **`mortredctl` 是真实 CLI，不是壳**：`src/apps/control/mortredctl.cpp` + `cli_app.cpp` 实现 11 个子命令（`init`、`doctor`、`upgrade`、`prepare`、`calibrate`、`init-trust`、`init-edge`、`install-host`、`next`、`status`、`catalog`），`cli_app.cpp:945-947` 与 `:1001` 是分派点；`scripts/mortredctl_*.sh` 提供同名包装（`mortredctl_doctor.sh`、`mortredctl_next.sh`、`mortredctl_init.sh`、`mortredctl_init-trust.sh`、`mortredctl_init-edge.sh`、`mortredctl_prepare.sh`、`mortredctl_upgrade.sh`、`mortredctl_calibrate.sh`、`mortredctl_install-host.sh`）。
- **`deploy/` 覆盖单机落地所需**：systemd 单元 `deploy/mortred-supervisor.service`（45 行，`ExecStart=/opt/mortred/bin/mortred-supervisor.out`、`Restart=always`、`RestartSec=3`、`KillMode=process`、`TimeoutStopSec=120`，`KillMode` 一行被 `scripts/stop_order_record.py:128-143` 当规则校验）；nginx 边缘（`nginx/mortred-edge.service`、`nginx.conf.skeleton`、`snippets/proxy-to-mortred.conf`、`ssl-params.conf`、`acme-challenge.conf`）；监控栈（`prometheus.yml` 76 行、`alert-rules.yml` 103 行、`grafana-dashboard.json` 113 行、`docker-compose.monitoring.yml`、`grafana/provisioning/**`）。
- **停止顺序有可执行规则检查**：编排方实跑 `python3 scripts/stop_order_record.py --check` → **退出码 0**，输出 4 行「为通过」（`KillMode=process`、`KillSignal=SIGTERM`、`TimeoutStopSec=120`、模型 SIGINT 并等待之后才向 gateway 发 SIGINT）加 3 行「不通过」——后三行是该脚本自身的**负例自检**（验证它能识别 gateway SIGINT 早于模型退出、未等两个模型退出、gateway 退出后 pid 复现），判据是退出码。
- **凭据生成与轮换有脚本**：`scripts/write_prometheus_credentials.sh`、`scripts/mortredctl_init-trust.sh`；`gateway_app.cpp:937` 的报错文案直接给出下一步命令（"Generate with: mortredctl init-trust"）。
- **活体证据（本项之外但在本模块内）**：P1-2 的实现者在提交版本上实跑 `MORTRED_API_TOKEN=… ./_bin/mortred-supervisor.out` + `./_bin/mortredctl.out --addr … status`，**退出码 0**，并观察到 supervisor 状态输出含每个 id 的 `state`/`pid`/`restarts`/`last_exit`/`error` 字段（见 `issues/p1-2/04-rd.md`）。
- 引擎门与占用门在第 9 轮已闭环（`scripts/test_p0_1_engine_fail_closed.sh`、`scripts/test_p0_2_unset_pack_doctor.sh` 是它们的黑盒回归脚本，`docs/benchmark-h3/` 提供 trtexec 参照）。

## 问题

无（本轮锁定的 7 个 P1 条目无一指向本模块的路径；`scripts/`、`deploy/`、`mortredctl` 在本轮未被修改）。

## 完成标准行

| 编号 | 判定 | 证据 / 缺什么 |
| --- | --- | --- |
| OP1（换机后能用现成命令为覆盖集每个 id 生成 engine） | **已满足** | 引擎生成链齐备：`scripts/build_trt_engine.sh`、`convert_trt_engines.sh`、`calibrate_pack.py`、`pack_trt.py`、`product_static_bs1.py`、`gen_trt_batch_profiles.py`、`trt_engine_builder.cc`；并有黑盒回归 `scripts/test_p0_1_engine_fail_closed.sh`（其头注释写明 AC8 需 GPU/trtexec、不可用时跳过） |
| OP2（`mortredctl status` / supervisor 状态接口能区分每个 id 的运行状态并看到最近一次退出信息） | **已满足** | `cli_app.cpp:1001` 的 `status` 子命令；P1-2 的活体冒烟输出含 `state`/`pid`/`restarts`/`last_exit`/`error`；`test/supervisor_unittest.cc` 覆盖状态与重启计数 |
| OP3（能只重启驻留集中的一个 id，不连带重启 gateway 或其他 id） | **已满足** | supervisor 管理 API 提供按 id 的 start/stop/restart（`supervisor_unittest` 的重启用例；P1-2 QA 的活体运行也逐 id 触发过 spawn）；**P1-3 落地后本行可再加强**（进程组回收保证孙进程不残留），但"单 id 重启"这一条本身已成立 |
| OP4（三类 token 缺失或彼此时相同仍能启动 = 未满足） | **已满足** | 见 `07-score-check.md` 的 OP4 行：`gateway_e2e_test.cc:969/1012/1020` 三个 e2e 用例 + `trust_tokens_unittest.cc:38,44` 的 `kScrapeTokenCollision` 断言 |
| OP5（收到停止信号后模型进程先退出；gateway 退出后不留下驻留集模型进程） | **部分：待 P1-3** | "gateway 最后退出/有序停机"已有 `supervisor_unittest.shutdown_is_ordered_and_complete`；"孙进程不残留"要等 **P1-3（进程组 + `process_group_gone`）** 合入后按整组回收重判 |
| TP3（缺 engine、缺占用戳或占用门闩被关闭时 `mortredctl doctor --strict` 不得退出 0；失败输出要给出下一条命令） | **阻塞** | 需要的验收是"在驻留集缺 engine/缺占用戳的状态下运行 `mortredctl doctor --strict`"；本机 `weights/` 是**被所有 worktree 目录联接共享**的实体，而 `scripts/test_p0_2_unset_pack_doctor.sh` 会移动/放置占位 engine（其头注释自述 "Moves or plants placeholders only"），在多个条目并行期间实跑有污染共享依赖树的风险，故本轮不跑。**恢复条件**：在没有并行工作占用 `weights/` 时，于仓库根运行 `env -u MORTRED_PACK bash scripts/test_p0_2_unset_pack_doctor.sh` 与 `bash scripts/test_p0_1_engine_fail_closed.sh`，记录退出码；随后核对 `weights/` 已还原（脚本内有 stash/restore） |

## 评分依据

- 给 8.0 而不是更高：① **TP3 未验证**（见上，属 engine/占用行，按 rubric 判阻塞而非已满足），而"缺 engine 时 doctor 必须拒绝"正是这台机器的关键安全线；② `scripts/` 有 70+ 个平铺文件、没有 `scripts/README` 索引，新人要从文件名猜用途（`__pycache__` 也留在树里）；③ 存在 `download_source_weights.ps1` 这类与 Ubuntu 目标平台不符的遗留物；④ 部分脚本是历史上一次性诊断（`verify_cls_scaffold_probe.py`、`check_legacy_yolov8s_engine_align.py`、`smoke_diffusion_async.sh`），未被任何 CI 或文档引用。
- 给 8.0 而不是更低：运维面**在同类项目里属于完备**——引擎生成/校准/打包成链、systemd 单元与停止顺序有可执行规则检查且**实测退出 0**、边缘与监控栈（prometheus/grafana/alert-rules）齐备、CLI 有 11 个真实子命令且活体可用、凭据生成与轮换有脚本、错误文案直接给出下一步命令。阻塞项只有一个且恢复条件明确。
- 参照系：如果 TP3 在本机跑通且 `weights/` 校验还原，本模块有理由到 8.5；反之若 `doctor --strict` 在缺 engine 时仍退出 0，则本模块应直接落到 6 以下（那正是完成标准把它写成门的原因）。
