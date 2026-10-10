# 分数核算

- 轮次: 11
- 原始综合分: 7.8
- 加权综合分: 7.8
- 上限规则: P1 上限 8.9
- 相对本轮起点: 8.5 到 7.8。销账不改分数。
- 出口: 继续
- 劣势集合: 相对本轮起点减少
- 挂起: P0-2（工作树缺 39 个产品 onnx，且工作树看不见 trtexec。未跑验收 6、7、8、12）。P1-8（msocrnet 在约 19GiB 的 WSL 上被 OOM 杀掉，没有黄金；工作树没有 full 构建的 model_golden_test。未跑验收 3 的 msocrnet 与 19 条同进程过滤，以及验收 11 的 full 列表）
- 与角色自算: 一致

模块分保持 `01-modules/` 的记录。加权仍是 776/100 = 7.76，保留一位小数为 7.8。P0-3 与 P1-10 已关闭。未关闭的 P0 只剩挂起的 P0-2，不触发 P0 上限。未关闭且未挂起的 P1 仍是 P1-11 与 P1-12，上限 8.9，不改变 7.8。

## 双目标
| 编号 | 状态: 已满足 / 未满足 / 挂起 | 证据或队列 id |
| --- | --- | --- |
| H1 | 已满足 | P1-7 合入 `8f6143544431ba6f7b9d64584cf7d1f6ff92fb28`。39 个 catalog path 各自有请求 example 和 200 example，并按该 path schema 校验。`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 分别是这三个 id。 |
| H2 | 未满足 | P1-8、P1-11、P1-12。`yolov8s` 已关闭的同类 IoU 门只覆盖 `yolov8s`。 |
| H3 | 已满足 | P1-9 合入 `5e39fdc28308254fa5c4c9feabbeb4a7d6ce730c`。表 5 有 39 行，每行同时有硬件、可执行命令、`trtexec_rps` 和日期。centerface 命令含 `--shapes=input.1:1x3x640x640`。 |
| H4 | 未满足 | P0-2（挂起，未合入）。不设 `MORTRED_PACK` 时，39 个 id 含 `yolov8s` 与 `metric3d_512`。 |
| C1 | 已满足 | P1-10 合入 `b17f70c502f76cb50325cae00c94def5e4888f50`。英文源码构建第一个 bash 围栏与中文路径 B 抽出的 GPU 集合都等于 `./scripts/install_deps.sh --all`、`sudo ./scripts/install_deps.sh --nvidia`、`./scripts/install_deps.sh --check`、`cmake --preset full && cmake --build --preset full`。两边仍有 `cmake --preset full` 与 `cmake --preset full-cpu`。 |
| C4 | 已满足 | `01-modules/ci.md`：`gen_openapi.py --check` 与 `check_contract_sync.py` 仍在。`gpu=skipped` 时 `inference-paths` 退出码 2 并打印 `NOT_A_TRT_PROOF`，不打印 `TRT_PROOF=yes`。 |
| S1 | 已满足 | P0-3 合入 `9b142453e686254a7ff9bab250a39cc826d7fc80`。`yolov7` 与 `yolov7x` 在输出非空且头不匹配时返回契约失败；空向量仍是 `model output empty`。`YoloV7OutputContract` 重跑退出码 0。已关闭的 SAM 原句不在此条。 |
| S2 | 未满足 | D-12 |
| S3 | 未满足 | D-8 |
| S4 | 未满足 | D-9。模型进程在 `allow()` 之前返回 401。未满足的是网关在鉴权前扣配额。 |
| S6 | 未满足 | D-10。`nvidia-smi` 整次失败且输出为空时已标 offline，不在此条。 |

D-11 不是双目标行，留在劣势队列。
