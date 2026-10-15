# 技术方案

- 问题: P2-17
- 版本: v1
- 对应列表: v3

## 边界

本方案只改英文与中文两份接口契约公共响应示例里 `model.name` 的示例值。一次打开 `docs/api-contract.md` 与 `docs/api-contract.zh-cn.md`。现示例都是 `"model": { "name": "MOBILENETV2", "version": "" }`。

对照物是 `docs/openapi.json` 里 `/mortred_ai_server_v1/classification/mobilenetv2` 的 200 响应 example：`"name": "mobilenetv2"`。本条不改 OpenAPI 本身（已正确）。

锁定的可观察变化（队列 v3）：一次打开这两份契约。公共响应示例的 `model.name` 都是 `mobilenetv2`。不升为 P0/P1。不扩到教程、how-to、代码、或 `docs/openapi.json`。

验收只打开文件。不拉起模型进程，不跑 GPU、calibrate 或压测。

## 方案

### 推荐

做法一：只把两份 api-contract 公共响应示例 JSON 里的 `"name": "MOBILENETV2"` 改成 `"name": "mobilenetv2"`，与 openapi 中 mobilenetv2 的 200 example 对齐。

**`docs/api-contract.md` 通用响应封装示例**由：

```json
  "model": { "name": "MOBILENETV2", "version": "" },
```

改为：

```json
  "model": { "name": "mobilenetv2", "version": "" },
```

**`docs/api-contract.zh-cn.md` 通用响应封装示例**同样由 `"name": "MOBILENETV2"` 改为 `"name": "mobilenetv2"`。

其余字段（`status`、`status_str`、`task_id`、`results`、`server_time_ms`、`partial`、422 示例）与两份契约中其它出现的小写 `mobilenetv2` 路径说明保持原样。只改这一处公共响应示例的 `name` 字段值，且只改这两份文件。

代价：对照过旧大写 `MOBILENETV2` 示例的笔记要对齐小写。没有进程、显存、engine 或压测代价。

### 备选

做法二：在旧大写示例旁加注「实际请用小写 mobilenetv2」，旧 `"name": "MOBILENETV2"` 仍保留。代价更小，但一次打开契约仍能读到与 openapi 不一致的示例，对不上锁定句。否决。

做法三：改 `docs/openapi.json` 把 mobilenetv2 的 200 example 改成大写 `MOBILENETV2` 以迁就契约。OpenAPI 已正确，且队列写明 openapi 本身 out of scope。否决。

做法四：连同两份教程、how-to、`README*`、产品代码、或其它 response 字段一起改。教程与 how-to 是别的开放项或已关项；本条只锁公共响应示例 `model.name`。否决。

## 改动面

只改 docs 模块里的两份接口契约。

- `docs/api-contract.md`：Common response envelope 示例中 `"name": "MOBILENETV2"` → `"name": "mobilenetv2"`。
- `docs/api-contract.zh-cn.md`：通用响应封装示例中 `"name": "MOBILENETV2"` → `"name": "mobilenetv2"`。

## 不改清单

保持原样的有：`docs/openapi.json`（对照物，mobilenetv2 200 example 已是 `"name": "mobilenetv2"`）；`docs/contract_dump.json`、`src/server/openapi_doc.h`；`docs/tutorials_of_model_servers.md`、`docs/tutorials_of_model_servers.zh-cn.md`；`docs/how_to_add_new_server.md`、`docs/how_to_add_new_server.zh-cn.md`；`README.md`、`README.zh-cn.md`；`docs/deployment.md`、`docs/deployment.zh-cn.md`；`docs/about_model_server_configuration.md` 及其中文；`docs/model-developer-guide.md`、`docs/model-developer-guide.zh-cn.md`；`docs/perf/coverage_set_http_rps.md`、`scripts/server/http_infer_rps.py`；`scripts/check_consistency.py`、`scripts/gen_openapi.py`。

代码与配置保持原样的有：`src/factory/classification_task.h`、`src/factory/cv_catalog.h`、各 task catalog；`conf/server/**`；产品 toml；`conf/packs/`；`src/apps/model_server_main.cpp`、`src/apps/product_index.h`；gateway（:8080）与 supervisor（:8787）的进程关系。每个 catalog id 仍是一个 OS 进程。每个 worker 仍持有各自的 TensorRT execution context。engine、static shape、batch=1、type=tensorrt 产品性能路径不改。运行时实际返回的 `model.name` 不因本条文档示例改动而改变；本条只对齐文档示例字符串。

## 契约

HTTP 请求、响应、OpenAPI 和 tensor 形状都不变。产品 toml 与 server toml 的键和值不变。变化只在两份 api-contract 公共响应示例 JSON 的 `model.name`：由 `MOBILENETV2` 对齐为与 openapi mobilenetv2 200 example 相同的小写 `mobilenetv2`。不新增或改写解析行为。

## 性能与资源

不生成 engine，不改 static shape、batch=1。每个 worker 仍各自持有自己的 TensorRT execution context，不增加共享 context 或跨 worker context 池。不改任何 `conf/server` 里的 `worker_nums`。不拉起 gateway、supervisor 或模型进程。不跑 `mortredctl prepare`、`mortredctl calibrate` 或 `http_infer_rps.py`。官方 HTTP 压测并发仍是已部署 `worker_nums` 的两倍，本条不改这条规则。显存与常驻进程无变化。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。cpu profile 仍不链 TensorRT。

## 验收标准

1. 一次打开 `docs/api-contract.md`。Common response envelope 示例出现 `"model": { "name": "mobilenetv2", "version": "" }`。不再出现 `"name": "MOBILENETV2"`。
2. 同一次打开 `docs/api-contract.zh-cn.md`。通用响应封装示例出现 `"model": { "name": "mobilenetv2", "version": "" }`。不再出现 `"name": "MOBILENETV2"`。
3. 打开对照：`docs/openapi.json` 中 `/mortred_ai_server_v1/classification/mobilenetv2` 的 200 example 仍含 `"name": "mobilenetv2"`。本条 diff 不触及 `docs/openapi.json`。
4. 本条 diff 只出现在 `docs/api-contract.md` 与 `docs/api-contract.zh-cn.md`。不触及两份教程、两份 how-to、产品/server 代码。

判定方式是打开文件。不启动 `mortred-model-server.out`，不跑 GPU，不跑 calibrate，不跑压测，不读 `/metrics`。

## 黑盒步骤

1. QA 在本机 WSL Ubuntu 工作目录执行下方自测命令（实现时用 worktree 根路径），期望退出码为 0。
2. QA 用 diff 确认本条改动只出现在 `docs/api-contract.md` 与 `docs/api-contract.zh-cn.md`；仓库里原有未跟踪的 ledger 不计入本条。
3. QA 按验收标准 1–4 打开文件核对。不拉起模型，不跑 calibrate，不跑压测。不把本条升为 P0/P1。`docs/openapi.json` 保持原样即通过。

## 自测命令

在本机 WSL Ubuntu 执行。工作目录为仓库根（基线或 worktree）。期望退出码为 0。断言失败时退出码为 1。本命令只读文件，不拉起 GPU，不跑 calibrate，不跑压测。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import pathlib
root = pathlib.Path("/mnt/g/Codex/mortred_model_server")
en = (root / "docs/api-contract.md").read_text(encoding="utf-8")
zh = (root / "docs/api-contract.zh-cn.md").read_text(encoding="utf-8")
oa = (root / "docs/openapi.json").read_text(encoding="utf-8")

old = '"name": "MOBILENETV2"'
new = '"name": "mobilenetv2"'
en_line = '  "model": { "name": "mobilenetv2", "version": "" },'
zh_line = '  "model": { "name": "mobilenetv2", "version": "" },'

assert en_line in en, "en missing aligned model.name example"
assert zh_line in zh, "zh missing aligned model.name example"
assert old not in en, "en still has MOBILENETV2 name"
assert old not in zh, "zh still has MOBILENETV2 name"
assert en.count(new) >= 1
assert zh.count(new) >= 1

assert '"name": "mobilenetv2"' in oa
assert "/mortred_ai_server_v1/classification/mobilenetv2" in oa
print("p2-17-api-contract-name ok")
PY
```

## RD 实现时注意

- 两份契约公共响应示例的 `model.name` 固定改为小写 `mobilenetv2`。不得只改英或只改中。
- 不得改 `version`、`task_id`、`results` 或其它示例字段。不得改 422 示例。
- 旧 `"name": "MOBILENETV2"` 必须从两份契约删除，不得旁注并存。
- 不改 `docs/openapi.json`。不改两份教程、how-to、README、产品代码。
- 不改 catalog() 真值、产品 toml、server toml、`model_server_main.cpp`、gateway(:8080)、supervisor(:8787)、TensorRT context 和 engine。一个 catalog id 仍是一个 OS 进程。
- 不改官方 HTTP 压测规则。本条的自测保持为上面的读文件命令。实现时将自测的 `cd`/`root` 换成 worktree 路径，断言正文不改。不把本条升为 P0/P1，不扩 scope。
