# 分数核算

- 轮次: 12
- 原始综合分: 7.8
- 加权综合分: 7.8
- 上限规则: 未关闭且未挂起的 P0、P1 都没有，不触发上限
- 相对本轮起点: 7.8 到 7.8。销账不改分数。
- 出口: awaiting-env
- 劣势集合: 相对本轮起点减少
- 挂起: P1-8（2026-10-11 `model_golden.msocrnet` 仍被杀掉，退出码 137，没有黄金。full 列表已跑过且退出码 0，`Yolov8TrtDetection` 仍在。未跑验收 3 的 msocrnet 与 19 条同进程过滤）。D-8（本机和 WSL 都没有可执行的 docker。未跑验收 1。未合入）
- 与角色自算: 一致

模块分保持第 11 轮记录。加权仍是 776/100 = 7.76，保留一位小数为 7.8。P0-2、P1-12、D-9、D-10、D-11 与 D-12 已关闭。非挂起劣势为无。未关闭的 P0 没有。未关闭的 P1 只剩挂起的 P1-8。挂起项不触发上限。H4 已满足。H2 与 S3 仍挂起。S1、S2、S4、S6 已满足。本机和 WSL 都没有可执行的 docker。WSL 可见内存约 19GiB，`msocrnet` 仍跑不了。环境恢复后只重跑挂起项的未跑验收。

## 双目标
| 编号 | 状态: 已满足 / 未满足 / 挂起 | 证据或队列 id |
| --- | --- | --- |
| H1 | 已满足 | P1-7 合入 `8f6143544431ba6f7b9d64584cf7d1f6ff92fb28`。39 个 catalog path 各自有请求 example 和 200 example，并按该 path schema 校验。`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 分别是这三个 id。 |
| H2 | 挂起 | 只剩 P1-8。2026-10-11 `model_golden.msocrnet` 仍被杀掉，退出码 137，`test/golden/msocrnet.json` 没有生成。`./build/full/test/model_golden_test --gtest_list_tests` 退出码 0，套件是 `model_golden.` 与 `Yolov8TrtDetection.`。未跑验收 3 的 msocrnet 与 19 条同进程过滤。P1-12 已关闭除 `yolov8s` 以外的同规则 TRT 比较。 |
| H3 | 已满足 | P1-9 合入 `5e39fdc28308254fa5c4c9feabbeb4a7d6ce730c`。表 5 有 39 行，每行同时有硬件、可执行命令、`trtexec_rps` 和日期。centerface 命令含 `--shapes=input.1:1x3x640x640`。 |
| H4 | 已满足 | P0-2 合入 `6389504638dccc00dbae60987c1c4b7fcfb55350`。不设置 `MORTRED_PACK` 时，冻结名单 39 个 id 各执行 `mortredctl prepare --id <id>`，退出码都是 0，表内 engine 存在且大小大于 0。 |
| C1 | 已满足 | P1-10 合入 `b17f70c502f76cb50325cae00c94def5e4888f50`。英文源码构建第一个 bash 围栏与中文路径 B 抽出的 GPU 集合都等于 `./scripts/install_deps.sh --all`、`sudo ./scripts/install_deps.sh --nvidia`、`./scripts/install_deps.sh --check`、`cmake --preset full && cmake --build --preset full`。两边仍有 `cmake --preset full` 与 `cmake --preset full-cpu`。 |
| C4 | 已满足 | `01-modules/ci.md`：`gen_openapi.py --check` 与 `check_contract_sync.py` 仍在。`gpu=skipped` 时 `inference-paths` 退出码 2 并打印 `NOT_A_TRT_PROOF`，不打印 `TRT_PROOF=yes`。 |
| S1 | 已满足 | P0-3 合入 `9b142453e686254a7ff9bab250a39cc826d7fc80`。`yolov7` 与 `yolov7x` 在输出非空且头不匹配时返回契约失败；空向量仍是 `model output empty`。已关闭的 SAM 原句不在此条。 |
| S2 | 已满足 | D-12 合入 `e8ff24a9d20944483b19b08f7022e99a8297017a`。同一条检查退出码 0，标准输出是 `fork-exec: pass`、`listen=127.0.0.1`、`MORTRED_AUTH_TOKEN=d12-auth`。 |
| S3 | 挂起 | 只剩 D-8。本机和 WSL 都没有可执行的 docker。未跑验收 1。未合入。 |
| S4 | 已满足 | D-9 合入 `b51e2e6e81ab8ac086ab39b33395cbe73ec28cc6`。`enabled = false` 且 `shadow = false` 时，无 Bearer 的 401 之后，紧接着的合法推理是 503 与 `service not ready`；同一秒内两次合法推理都是这条 503；scope 为 `none` 的 key 连续两次都是 401。 |
| S6 | 已满足 | D-10 合入 `9877ac70a6cda2ce88cfef1c09670789e21f6300`。utilization 列为 `N/A` 或 `[N/A]` 的这一次探测，`available` 为 false，stderr 是 `gpu sampler offline (utilization N/A)`，`mortredctl ps` 印 `util n/a`。空输出仍是原来的 `gpu sampler offline (no nvidia-smi / no GPU)`。 |

D-11 已关闭，不是双目标行。非挂起队列已空。出口是 awaiting-env。
