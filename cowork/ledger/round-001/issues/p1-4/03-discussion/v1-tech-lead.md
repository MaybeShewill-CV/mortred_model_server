# 技术方案

- 问题: P1-4 文档里网关路径与任务响应形态分裂，每次接客户 SDK 易接错
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` P1-4；上一版审查意见无（本条首版）

## 边界

- 只改必读四份：`docs/api-contract.md`、`docs/api-contract.zh-cn.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`。中英同一口径。
- 不改运行时 HTTP（gateway、模型端口、`server_uri` 仍按现网工作）。不改 P0，不做 P1-3。不启服务、不训模型、不碰非 Linux。
- 推荐客户端入口必须是已有 **`POST /v1/models/{id}/infer`**（gateway `:8080`）。`{id}` 用现有 catalog，如 `yolov8s`、`mobilenetv2`、`pphuman_server`、`dbnet`。
- `/mortred_ai_server_v1/...` 与 toml 里的 `server_uri` 若仍出现，只能标成**遗留/兼容**，不得再当架构图或端口表的推荐入口。
- `results[].data` 字段必须与现行 `docs/openapi.json` `components.schemas` 及 `src/server/response_serializers.h` 一致；OCR 只给文字检测（`TextRegionResult`），不写识别文本。

## 做法

### 推荐

**必读文档收口：推荐路径写死 + 四类任务 payload 人读示例进契约页。**

在 `deployment` 架构图（客户端→GW 那条边）和 `:8080` 端口表把推理入口改成 `POST /v1/models/{id}/infer`；另用一句标明遗留 `{server_uri}` / `/mortred_ai_server_v1/...` 仍被 gateway 接受，但新 SDK 不要走。在 `api-contract` 通用信封之后增加「按任务的 `results[].data`」节，各给一段可核对 JSON（检测数组、分类对象、分割对象、OCR 检测数组）；分类可与现有信封示例共用字段，但任务节必须能单独核对。机器可读契约仍指向 `docs/openapi.json`，人读页不再把任务字段只推到 OpenAPI/序列化头。

代价：文档维护，中英四处同步；不增加进程、显存、engine 工作。联调摩擦按冻结口径可黑盒关掉。

### 否决

1. **改 gateway 只认 `/v1/...`、去掉遗留 `server_uri`。** 这是运行时 HTTP，越界；旧脚本会断。本条只解决「docs 教错入口」。
2. **把 `docs/openapi.json` 路径从模型口 `server_uri` 改成 `/v1/models/{id}/infer`，或新开 sdk cookbook 当主入口。** OpenAPI 仍是模型进程路径，改了会和真实 `{server_uri}` 假对齐且易被当成契约变更；新文件不在必读四份里，验收「在必读 docs 内核对」落空。FAQ 破损 `</details>`、GPU pack 步骤、README 不在本条。

## 改动面

- `docs/deployment.md` / `docs/deployment.zh-cn.md`：§1 mermaid 客户端边、`:8080` Purpose 列；需要时加一行「推荐 / 遗留」对照。§15 FAQ 里已有的 `/v1/models/{id}/jobs` 保持，并与推荐 `/infer` 一致（jobs 仍非本条主路径）。
- `docs/api-contract.md` / `docs/api-contract.zh-cn.md`：拓扑里写清推荐 `POST /v1/models/{id}/infer`；遗留 `{server_uri}` 保留现有路由表但标明非推荐；补四类 `results[].data` 示例。现有「详见 openapi / serializers」改为「人读示例如下，完整 schema 仍见 OpenAPI」。

## 明确不改

- `src/`、`conf/`、`scripts/`、gateway/supervisor 行为、产品 toml 的 `type=`、engine/`mortredctl prepare`。
- `docs/openapi.json`、`scripts/gen_openapi.py`、`src/server/response_serializers.h`。
- P0-1/P0-2、P1-1～P1-3、README、`docs/async-jobs-*`、`docs/perf/*`、测试二进制。
- 信封顶层字段（`status` / `results[]` / 禁止 `img_data`）与 HTTP 状态映射。

## 契约

- **HTTP/tensor 运行时：不变。** 推荐路径本来就能打到现网 gateway。
- **文档契约（本条交付）：**
  - 推荐：`POST http://<gateway>:8080/v1/models/{id}/infer`，Bearer 与现契约相同。
  - 遗留：`POST {server_uri}`（形如 `/mortred_ai_server_v1/...`）仍可用，文档不得再画成默认箭头。
  - 请求信封不变：`images[]` 等。
  - `results[].data` 人读示例必须能对上（数值可示意）：
    - 分类：`{"class_id": 123, "category": "tabby cat", "scores": [0.1, 0.8, 0.1]}`
    - 检测：数组元素含 `class_id`、`score`、`category`、`bbox`（`[x1,y1,x2,y2]`）、`detail_infos`
    - 分割：`{"image": "<base64>", "colorized_mask": "<base64>"}`（选分割作为「分割或抠图」代表；不得把抠图写成带 `colorized_mask`）
    - OCR 检测：数组元素含 `score`、`bbox`、`polygon`、`detail_infos`；**禁止** `text` / 识别字段
  - 路径示例里的 `{id}` 必须是真实 catalog（上列四个），不要写已删除的 `YOLOV8` 这类旧名。

## 性能与资源

- 常驻进程树不变：一 catalog 一 OS 进程；gateway `:8080`、supervisor `:8787`。
- 无新增显存、无新 engine、无 cpu/gpu profile 变化。文档改完不要求重跑 `prepare`。

## 验收标准

均在仓库内只读文档、用 `rg`、不启服务即可判定。

- **A1** `docs/deployment.md` 与 `docs/deployment.zh-cn.md` 的架构 mermaid 中，客户端到 gateway 的标注含 `POST /v1/models/{id}/infer`（或等价的 `/v1/models/{id}/infer` 且方法为 POST）。
- **A2** 上述两文件 `:8080` 端口用途把 `POST /v1/models/{id}/infer` 列为推理推荐入口，而不是未加遗留说明的 `/mortred_ai_server_v1/...`。
- **A3** 四份必读里若仍出现 `/mortred_ai_server_v1`，同一文件须同时出现 legacy 或「遗留」；mermaid 客户端边不得再指向 `/mortred_ai_server_v1`。
- **A4** 两份 `api-contract` 均含可核对的 `results[].data` 示例，且同时覆盖：分类（`class_id`+`category`+`scores`）、检测（`bbox`+`detail_infos` 的数组项）、分割（`image`+`colorized_mask`）、OCR 检测（`polygon` 且无识别字段名 `text` 出现在该 OCR 示例块）。
- **A5** 两份 `api-contract` 均把 `POST /v1/models/{id}/infer` 写为经 gateway 的推荐推理路径。
- **A6** 相对本条工作区，`src/`、`conf/`、`docs/openapi.json` 无改动。

## 测试义务

QA 白盒只对文档与 git 路径，不启 `:8080`/`:8787`、不 `curl` 推理。

- 执行下方自测命令，退出码须全部符合。
- 抽读 A4 四段 JSON：类型（对象 vs 数组）与必填键与 OpenAPI `ClassificationResult` / `DetectionResult` / `SegmentationResult` / `TextRegionResult` 一致。
- 确认 OCR 示例不是识别结果；分割示例未与抠图 `MattingResult`（仅 `image`）混用。
- 中英四文件对 A1–A5 对等，无一份仍把遗留路径当推荐。
- 不测 P0、P1-3，不测 live infer、engine、systemd。

## 自测命令

在仓库根 `/mnt/g/Codex/mortred_model_server` 执行。`rg` 有匹配退出码 **0**，无匹配 **1**。

```bash
# A1+A5：四份必读均出现推荐路径（期望退出码 0）
rg -F --files-with-matches 'POST /v1/models/{id}/infer' \
  docs/api-contract.md docs/api-contract.zh-cn.md \
  docs/deployment.md docs/deployment.zh-cn.md | wc -l
# 期望打印 4；然后：
test "$(rg -F --files-with-matches 'POST /v1/models/{id}/infer' \
  docs/api-contract.md docs/api-contract.zh-cn.md \
  docs/deployment.md docs/deployment.zh-cn.md | wc -l)" = 4
```

期望退出码 **0**。

```bash
# A1：mermaid 客户端边不再指向遗留路径（rg 无匹配，再用 test 把 1 转成 0）
rg -q 'C .*mortred_ai_server_v1' docs/deployment.md docs/deployment.zh-cn.md
test "$?" -eq 1
```

期望退出码 **0**。

```bash
# A1：mermaid 指向推荐路径（期望 rg 退出码 0）
rg -q 'C .* /v1/models/' docs/deployment.md
rg -q 'C .* /v1/models/' docs/deployment.zh-cn.md
```

期望两句均为 **0**。

```bash
# A4：契约页具备四类字段（期望每条 rg 退出码 0）
rg -q '"class_id"' docs/api-contract.md docs/api-contract.zh-cn.md
rg -q '"scores"' docs/api-contract.md docs/api-contract.zh-cn.md
rg -q '"bbox"' docs/api-contract.md docs/api-contract.zh-cn.md
rg -q '"detail_infos"' docs/api-contract.md docs/api-contract.zh-cn.md
rg -q '"colorized_mask"' docs/api-contract.md docs/api-contract.zh-cn.md
rg -q '"polygon"' docs/api-contract.md docs/api-contract.zh-cn.md
```

```bash
# A3：deployment 若仍写 mortred_ai_server_v1，则必须带 legacy 或 遗留（人工对照 rg 上下文；脚本门：两文件各自同时命中或同时不命中路径词与遗留词）
rg -n 'mortred_ai_server_v1' docs/deployment.md docs/deployment.zh-cn.md \
  docs/api-contract.md docs/api-contract.zh-cn.md
rg -n -i 'legacy|遗留' docs/deployment.md docs/deployment.zh-cn.md \
  docs/api-contract.md docs/api-contract.zh-cn.md
```

有遗留路径命中时，同文件须有 legacy/遗留命中；QA 按 A3 判。若四文件已无该路径，前一条 rg 退出码 **1** 可接受。

```bash
# A6：运行时与 OpenAPI 未改（无 diff 时期望退出码 0）
test -z "$(git diff --name-only -- src conf docs/openapi.json)"
```

## RD 不得自行决定

- 不得把推荐入口改成 `{server_uri}`、模型直连端口或 `/v1/models/{id}/jobs`。
- 不得改 C++/gateway/OpenAPI 生成来「顺便统一路径」。
- 不得自造 `results[].data` 字段；OCR 不得加识别结果。
- 不得只改英文或只改 `api-contract`；四份必读必须一起交。
- 不得用本条修 FAQ 结构、GPU pack、P1-3 安装或任何 P0。
- `{id}` 示例不得使用已废弃大写 HTTP 名。
