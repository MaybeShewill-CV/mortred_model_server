# 技术方案

- 问题: P0-2
- 版本: v2
- 对应列表: `/mnt/g/Codex/mortred_model_server/cowork/ledger/round-001/02-p0-p1-final.md`

## 边界

- 只收口：scrape token 配错 / 网关未就绪时的 **可观测永久失败**，以及 autostart **勿继续起 model** 的终态；使 `GET /api/v1/status` 与 `mortredctl doctor --live` 能当场定位，不再只靠 stderr。
- 守住：一 catalog id 一进程；gateway `:8080`、supervisor `:8787`；后端仍为 MNN / ONNX Runtime / TensorRT；产品 toml 保持既有 `type` 约定；CUDA 12.x + TensorRT 10.x；engine 本机本卡 `prepare`；cpu profile 不链 TensorRT。
- 不推翻 P0-1：`mortredctl doctor` 对缺 pack engine / occupancy stamp 的默认 fail-closed 保持；本条只在 live 路径上 **叠加** 网关/autostart 诊断，不得把该门禁改回 WARN-only。
- 不改：训练与标注、非 Linux、任一 P1、模型推理路径、网关对外推断契约本体。

## 做法

### 推荐

**网关先决 + 失败可查：supervisor 管理面保持可访问；scrape token 永久失败或网关在 deadline 内未就绪时，把原因写入 gateway 状态（及 status 上的 autostart 阻断字段），本次 autostart 不再拉起任何 model；`doctor --live` 优先读 supervisor status 打印该终态并非 0 退出。**

代价：改 `ProcessSupervisor::autostart_all`、软化 `SupervisorApp::init` 对 scrape 的「进程直接拒启」、扩展 `/api/v1/status` 只读字段、改 `mortredctl_doctor.sh` / live 验收脚本与对应单测。不改模型二进制、不改 `:8080` 推断鉴权规则本身。换 token / 首次上机时可在 status 与 doctor 看到「为何无路由、为何 model 未起」。

### 否决

1. **只加 stderr / 日志，不拦 model、不改 status/doctor**
   代价最低，但达不到冻结条目「勿继续起 model」与「不再只靠翻 stderr」。

2. **保持 init 对空/碰撞 scrape 直接 `return false`（supervisor 不起、API 不可达），只增强 doctor 静态读环境变量**
   能覆盖「token 未配」的静态提示，但冻结验收要求 **supervisor 状态/API/`doctor --live`**；token 案下管理面起不来时 live 只能回退 basic，仍难对上「永久失败终态」黑盒。

## 改动面

- **`ProcessSupervisor::autostart_all`（`supervisor.cpp`）**
  - 网关阶段任一下列情况视为 **本次 autostart 阻断**，**不得**进入 model 启动循环：gateway 二进制缺失；`start_server(__gateway)` 失败（含 scrape 永久失败）；deadline 内 `status(__gateway).state != "running"`（含仍 `starting` / 已 `failed`）。
  - 阻断时：把明确原因写入 `__gateway` 的 `error`（沿用/补充现有 scrape / deadline 文案）；保证可从 `status()` / `statuses()` 读到；model 侧保持未 wanted / 未 spawn（本次波次零 model 启动）。
  - stderr 可保留，但不得再是唯一信息源。
- **`SupervisorApp::init` scrape 早退（`supervisor_app.cpp`）**
  - 存在 `mortred-gateway.out` 且 scrape 不可用时：**不再**因 scrape 使整个 supervisor 进程 init 失败退出。
  - 改为：照常创建 supervisor、listen、再走 `autostart_all`；由 spawn/permanent-failure + 上述阻断逻辑把 `__gateway` 置于 `failed`（或等价终态）并跳过 model。
  - 既有单测 `scrape_required_when_gateway_binary_present` 须按本语义改写为「init/listen 成功 + gateway failed + models 未起」。
- **`GET /api/v1/status`（只读扩展）**
  - 在现有 `gateway` / `servers[]` 之外增加可黑盒核对的 autostart 阻断摘要（字段名由 RD 实现，但语义固定），至少能判定：
    - 本次 autostart 是否因网关未就绪/永久失败而 **跳过全部 model**；
    - 人类可读 `reason`（须包含 scrape 原因关键字，或 gateway deadline / binary missing 等对应原文）。
  - `gateway.state` / `gateway.error` 在 scrape 永久失败时须继续为 `failed` + 含 `gateway scrape token rejected:`（或与 `trust_tokens.h` 一致的既有文案）。
- **`mortredctl doctor` live 路径（`scripts/mortredctl_doctor.sh`，必要时薄改 `verify_deployment.sh --live`）**
  - supervisor `/api/v1/health` 可达后：**先**拉取 `/api/v1/status`（或 CLI 等价），若 gateway 永久失败或 autostart 已阻断起 model，打印 `gateway.error` / 阻断 `reason`，**退出码 ≠ 0**，并明示勿指望无路由下的 model 已起。
  - 仅当 gateway 已 `running` 且未阻断时，再跑现有 gateway `/healthz`、推断 401、`/metrics` 401 等 live 探针。
  - **不得**削弱 P0-1 已合入的 pack TRT / occupancy 默认 fail-closed。
- **测试**
  - 扩展 `test/supervisor_unittest.cc`（及必要时 `supervisor_multiinstance_test.cc`）：覆盖「gateway scrape 永久失败 → 不启 model」「gateway deadline 未就绪 → 不启 model」「status 阻断字段可读」。
  - 保留既有 `gateway_missing_scrape_token_is_permanent_failure` / collision 语义。

## 明确不改

- P0-1：`doctor`/`init` 对缺 engine/stamp（及 init 权重失败）的默认非 0；`product_static_bs1` 真源。
- 为「无 `uri` 的任意 catalog child」全面补 readiness（冻结「改完后」以 token/网关与勿起 model 为准；gateway 继续用既有 `/healthz`）。
- 一 id 一进程、`:8080` / `:8787`、MNN/ORT/TRT 后端集合、产品 toml `type` 约定。
- scrape 安全规则本身：metrics token 仍须非空且不得与 infer/admin 相同（`trust_tokens.h`）；不得改为「空 token 仍起 gateway」。
- 训练、标注、非 Linux、任一 P1（OCR 文档口径、pack 校准样本、主机 install 收口、docs 路径统一）。

## 契约

- **HTTP 管理面**：`GET /api/v1/status` 增加只读 autostart 阻断摘要（见上）；既有 `gateway`/`servers` 字段语义保持；不改 gateway 推断路径与任务信封。
- **进程**：scrape 配错时 supervisor **进程可存活**（`:8787` 可查）；`__gateway` 为永久 `failed`（或 deadline 阻断下 error 已写入且本次未宣称成功）；本次 autostart **零 model spawn**。
- **命令**：`mortredctl doctor`（live：supervisor 可达）在上述阻断/永久失败下退出码 ≠ 0，输出含失败原因关键字；pack engine/stamp 门禁行为与 P0-1 一致。
- **token**：`MORTRED_METRICS_TOKEN` 空或与 infer/admin 碰撞 → 仍拒绝起 gateway；原因须出现在 status/API/doctor，而非仅 stderr。

## 性能与资源

- 不改常驻进程模型（仍一 id 一进程 + 隐式 gateway）；阻断时 **少** 起 model，显存与 engine 负载只降不升。
- 不改 TRT engine 生成/精度；不新增常驻探测频率要求（沿用现有 gateway `/healthz` 与 deadline）。
- cpu/gpu profile 分流不因本条改变。

## 验收标准

1. **AC1（空 scrape → 永久失败可查，且不起 model）**：在存在 `mortred-gateway.out`、清空 `MORTRED_METRICS_TOKEN`（或等价未设置）的前提下启动 supervisor；进程须监听管理面；`GET /api/v1/status` 中 `__gateway`（或 `gateway`）为 `failed`，`error` 含 scrape/metrics token 相关原因；autostart 阻断摘要表明已跳过 model；`servers` 中本次 pack 内 model 均未进入 `starting`/`running`（pid 均不可为存活的本次 autostart 产物）。
2. **AC2（碰撞 scrape → 同上）**：`MORTRED_METRICS_TOKEN` 与推理或管理 token 相同；行为同 AC1，`error` 含 collision/matches 类原因。
3. **AC3（网关 deadline 未就绪 → status/doctor 可查阻断，且不起 model）**：scrape 合法但网关在 gateway deadline 内未到 `running`；autostart 结束后：`GET /api/v1/status` 须给出人类可读的失败/阻断原因（`gateway.error` 和/或 autostart 阻断 `reason`，须含 deadline / 未就绪类关键字）；任一次 autostart 队列中的 model 不得因该次 `autostart_all` 被启动；supervisor 可达时 `mortredctl doctor`（live，不必 `--strict`）退出码 ≠ 0，输出含上述同一可读原因，并能看出本次未起 model。不得只验「不起 model」。
4. **AC4（doctor --live 表面终态）**：在 AC1 或 AC2 场景下，supervisor 可达时执行 `mortredctl doctor`（走 live，不必 `--strict`），退出码 ≠ 0，stdout/stderr 出现 gateway 永久失败原因（与 status `error` 可对上），且能看出 model 未因该次启动而就绪；不得仅报「start the service first」或只给 `/healthz` 探针失败而无 token/阻断原因。
5. **AC5（P0-1 不回退）**：当前 pack 缺 engine 或必需 occupancy stamp 时，无 `--strict` 的 `mortredctl doctor` 仍退出码 ≠ 0 并点名缺失物（与 P0-1 已合入行为一致）；本条改动不得使其变绿。
6. **AC6（正常路径不误伤）**：scrape 合法且网关在 deadline 内 `running` 时，autostart 仍可按既有策略拉起 pack 内 model；status 阻断摘要不得误报「已跳过 model」。

## 测试义务

- **QA（黑盒）**：逐条 AC1–AC6；至少覆盖空 token、碰撞 token、gateway deadline 未就绪时的 status 可读原因 + doctor live 非 0 与原因字样 + 本次未起 model；并回归 AC5（P0-1 doctor）。
- **白盒**：核对 `autostart_all` 在网关失败/deadline 后无 model `start_server`；init 不再因 scrape 单独杀死管理面；`/api/v1/status` 字段与 `trust_tokens` 文案一致。
- **回归边界**：不测 P1；不测「给全部无 uri model 补 ready」；不测非 Linux；不测训练。

## 自测命令

在仓库根目录执行（build 目录按本机已有 preset，下表以 `build-ci` 为例，可换成实际 build dir）：

| 目的 | 命令 | 期望退出码 |
| --- | --- | --- |
| scrape token 单测 | `bash scripts/run_tests.sh build-ci -R trust_tokens_unittest --output-on-failure` | `0` |
| supervisor 单测（含本条新增/改写用例） | `bash scripts/run_tests.sh build-ci -R 'supervisor_unittest|SupervisorAppInit' --output-on-failure` | `0` |
| 空 token 起管面后看 status | （清空 `MORTRED_METRICS_TOKEN`，保留 gateway 二进制，启动 supervisor 后）`curl -fsS "$MORTREDCTL_ADDR/api/v1/status"`（或带管理 token 的等价请求） | curl `0`；JSON 中 gateway `failed` + scrape 原因，且阻断跳过 model |
| doctor live（AC1/AC4） | （同上 supervisor 仍在跑）`bash scripts/mortredctl_doctor.sh` | `≠0`；输出含 scrape/永久失败原因 |
| doctor live（AC3 deadline） | （scrape 合法、人为使 gateway 在 deadline 内达不到 `running`，supervisor 仍可达）先 `curl` status，再 `bash scripts/mortredctl_doctor.sh` | curl `0` 且 JSON 含 deadline/未就绪原因并跳过 model；doctor `≠0`，输出含同一原因 |
| doctor 不回退 P0-1（AC5） | `MORTRED_PACK="$PWD/conf/packs/yolov8.toml" bash scripts/mortredctl_doctor.sh`（在缺 engine 或缺 stamp 前提下） | `≠0`；输出点名 engine 或 stamp |
| 正常 scrape 冒烟（有环境时） | scrape 合法且 gateway/pack 就绪后 `bash scripts/mortredctl_doctor.sh` | `0`（在 P0-1 门禁与 live 探针均绿时） |

## RD 不得自行决定

- scrape 配错时是否允许 supervisor 管理面存活以便 status/API/doctor 观测（本方案：**必须存活**；不得恢复为「init 直接退出且无 API」作为唯一终态）。
- 网关失败/未就绪后是否继续 autostart model（本方案：**禁止**本次继续）。
- 是否削弱 P0-1 doctor 对缺 engine/stamp 的默认 fail-closed（**否**）。
- 是否把范围扩到全 catalog「无 uri 补 readiness」或任一 P1（**否**）。
- 是否放宽 `trust_tokens`（空 metrics 或与 infer/admin 共用仍须拒绝起 gateway）。
