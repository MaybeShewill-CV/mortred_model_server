# 技术方案

- 问题: P1-4 文档里网关路径与任务响应形态分裂，每次接客户 SDK 易接错
- 版本: v2
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` P1-4；吸收 v1 PM 条件（A1 删「或等价」、A4 OCR 只看 JSON 键名）与 v1 RD 条件（自测按文件分别判定、A2 端口表可复现、A4 四段 JSON 分别核对且检测不得用 OCR 冒充）

## 边界

- 只改必读四份：`docs/api-contract.md`、`docs/api-contract.zh-cn.md`、`docs/deployment.md`、`docs/deployment.zh-cn.md`。中英同一口径。
- 不改运行时 HTTP（gateway、模型端口、`server_uri` 仍按现网工作）。不改 P0，不做 P1-3。不启服务、不训模型、不碰非 Linux。
- 推荐客户端入口必须是已有 **`POST /v1/models/{id}/infer`**（gateway `:8080`）。`{id}` 用现有 catalog，如 `yolov8s`、`mobilenetv2`、`pphuman_server`、`dbnet`。
- `/mortred_ai_server_v1/...` 与 toml 里的 `server_uri` 若仍出现，只能标成**遗留/兼容**，不得再当架构图或端口表的推荐入口。
- `results[].data` 字段必须与现行 `docs/openapi.json` `components.schemas` 及 `src/server/response_serializers.h` 一致；OCR 只给文字检测（`TextRegionResult`），不写识别文本。
- 测试只在 WSL 用所列 `rg` / `test` / `python3` 对仓库内文档做只读核对，不启 `:8080`/`:8787`，不 `curl` 推理。

## 做法

### 推荐

**必读文档收口：推荐路径写死 + 四类任务 payload 人读示例进契约页。**

在两份 `deployment` 架构图（客户端→GW 那条边）的标注里写入字面 `POST /v1/models/{id}/infer`；`:8080` 端口表用途列同样写入该字面量作为推理推荐入口。另用一句标明遗留 `{server_uri}` / `/mortred_ai_server_v1/...` 仍被 gateway 接受，但新 SDK 不要走。在 `api-contract` 通用信封之后增加「按任务的 `results[].data`」节，各给一段可单独解析的 JSON（检测数组、分类对象、分割对象、OCR 检测数组）；分类可与现有信封示例共用字段，但任务节必须能单独核对。机器可读契约仍指向 `docs/openapi.json`，人读页不再把任务字段只推到 OpenAPI/序列化头。

代价：文档维护，中英四处同步；不增加进程、显存、engine 工作。联调摩擦按冻结口径可黑盒关掉。

### 否决

1. **改 gateway 只认 `/v1/...`、去掉遗留 `server_uri`。** 这是运行时 HTTP，越界；旧脚本会断。本条只解决「docs 教错入口」。
2. **把 `docs/openapi.json` 路径从模型口 `server_uri` 改成 `/v1/models/{id}/infer`，或新开 sdk cookbook 当主入口。** OpenAPI 仍是模型进程路径，改了会和真实 `{server_uri}` 假对齐且易被当成契约变更；新文件不在必读四份里，验收「在必读 docs 内核对」落空。FAQ 破损 `</details>`、GPU pack 步骤、README 不在本条。

## 改动面

- `docs/deployment.md` / `docs/deployment.zh-cn.md`：§1 mermaid 客户端→GW 边标注必须含字面 `POST /v1/models/{id}/infer`；`:8080` Purpose/用途列必须含同一字面量。需要时加一行「推荐 / 遗留」对照。§15 FAQ 里已有的 `/v1/models/{id}/jobs` 保持，并与推荐 `/infer` 一致（jobs 仍非本条主路径）。
- `docs/api-contract.md` / `docs/api-contract.zh-cn.md`：拓扑里写清推荐 `POST /v1/models/{id}/infer`；遗留 `{server_uri}` 保留现有路由表但标明非推荐；补四类 `results[].data` 示例，且每类必须是独立可解析的 JSON 代码块。现有「详见 openapi / serializers」改为「人读示例如下，完整 schema 仍见 OpenAPI」。

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
  - 两份 `deployment` 架构图里客户端到 gateway 的边，标注必须含字面 `POST /v1/models/{id}/infer`。不得用「仅路径、方法另写」或其它「或等价」写法替代。
  - 两份 `deployment` 的 `:8080` 用途列必须含字面量 `POST /v1/models/{id}/infer`。该行若仍出现 `/mortred_ai_server_v1`，同一行必须同时出现 `legacy` 或「遗留」，不得把未标明遗留的 `/mortred_ai_server_v1` 当作推荐推理入口。
  - `results[].data` 人读示例必须能对上（数值可示意），且四类必须是四段独立 JSON（对象或数组），不得互相冒充：
    - 分类：对象含 `class_id`、`category`、`scores`
    - 检测：数组项含 `bbox` 与 `detail_infos`（该数组项不得含 `polygon`，以免 OCR 块同名字段冒充）
    - 分割：对象含 `image` 与 `colorized_mask`（选分割作为「分割或抠图」代表；不得把抠图写成带 `colorized_mask`）
    - OCR 检测：数组项含 `polygon`；该示例对象/数组元素不得出现键 `text`。A4 的 OCR 判定只看 JSON 示例里的键名：该示例对象/数组元素不得出现键 `text`；正文写「禁止 text / 识别字段」不算失败。
  - 路径示例里的 `{id}` 必须是真实 catalog（上列四个），不要写已删除的 `YOLOV8` 这类旧名。

## 性能与资源

- 常驻进程树不变：一 catalog 一 OS 进程；gateway `:8080`、supervisor `:8787`。
- 无新增显存、无新 engine、无 cpu/gpu profile 变化。文档改完不要求重跑 `prepare`。

## 验收标准

均在仓库内只读文档、用下方自测命令、不启服务即可判定。一条命令失败即该条未过，不得用「扫两份文件一次命中」代替按文件判定。

- **A1** `docs/deployment.md` 与 `docs/deployment.zh-cn.md` 各自 §1 架构 mermaid 中，客户端到 gateway 的边标注必须含字面 `POST /v1/models/{id}/infer`。不含该字面即失败。不得用「或等价」口径（例如只写 `/v1/models/{id}/infer`、方法另注 POST）判过。
- **A2** 上述两文件各自端口表中端口为 `8080` 的那一行，用途列含字面量 `POST /v1/models/{id}/infer`；且该行不得把未标明遗留的 `/mortred_ai_server_v1` 当作推荐推理入口（该行若出现 `/mortred_ai_server_v1`，同一行必须含 `legacy` 或「遗留」）。
- **A3** 四份必读里若仍出现 `/mortred_ai_server_v1`，同一文件须同时出现 `legacy` 或「遗留」；mermaid 客户端边不得再指向 `/mortred_ai_server_v1`。
- **A4** 两份 `api-contract` **分别**含可核对的四段 `results[].data` JSON，且同时覆盖、互不冒充：分类对象含 `class_id`/`category`/`scores`；检测为含 `bbox`+`detail_infos` 的数组项（不得用 OCR 块里的同名字段冒充）；分割对象含 `image`+`colorized_mask`；OCR 检测块含 `polygon` 且该块不出现字段名 `"text"`。OCR 只看该 JSON 示例对象/数组元素的键名：不得出现键 `text`；正文写「禁止 text / 识别字段」不算失败。
- **A5** 两份 `api-contract` **分别**把 `POST /v1/models/{id}/infer` 写为经 gateway 的推荐推理路径（每文件单独命中该字面量）。
- **A6** 相对本条工作区，`src/`、`conf/`、`docs/openapi.json` 无改动。

## 测试义务

QA 白盒只对文档与 git 路径，不启 `:8080`/`:8787`、不 `curl` 推理。在 WSL 仓库根执行下方命令，退出码须全部符合。

- 执行下方自测命令；任一文件、任一段 JSON 失败即整条失败。
- 抽读 A4 四段 JSON：类型（对象 vs 数组）与必填键与 OpenAPI `ClassificationResult` / `DetectionResult` / `SegmentationResult` / `TextRegionResult` 一致。检测项不得拿 OCR 数组项顶替。
- 确认 OCR 示例不是识别结果（键名无 `text`）；分割示例未与抠图 `MattingResult`（仅 `image`）混用。
- 中英四文件对 A1–A5 对等，无一份仍把未标明遗留的路径当推荐。
- 不测 P0、P1-3，不测 live infer、engine、systemd。

## 自测命令

在仓库根 `/mnt/g/Codex/mortred_model_server` 执行。每条命令独立；失败退出码 **1**，通过 **0**。不得把两份 `api-contract` 或两份 `deployment` 合成一次 `rg -q` 扫过就算过。

```bash
# A1：两份 deployment 架构图客户端→GW 边含字面 POST /v1/models/{id}/infer
# 期望退出码 0；任一文件缺字面则 1
python3 - <<'PY'
from pathlib import Path
import re, sys
pat = re.compile(r'C\s*-->\|"([^"]*)"\|\s*GW')
need = "POST /v1/models/{id}/infer"
ok = True
for p in ["docs/deployment.md", "docs/deployment.zh-cn.md"]:
    text = Path(p).read_text(encoding="utf-8")
    m = re.search(r"```mermaid\n(.*?)```", text, re.S)
    if not m:
        print(f"FAIL {p}: no mermaid"); ok = False; continue
    labels = pat.findall(m.group(1))
    if not any(need in lab for lab in labels):
        print(f"FAIL {p}: C→GW label missing literal {need!r}; got {labels!r}")
        ok = False
    else:
        print(f"OK {p}: C→GW has {need!r}")
sys.exit(0 if ok else 1)
PY
```

期望退出码 **0**。

```bash
# A1：mermaid 客户端边不得指向遗留路径（按文件分别判定）
# 每文件：有 C→…mortred_ai_server_v1…→GW 则失败（rg 命中期望 1，再 test 转 0）
rg -q 'C .*mortred_ai_server_v1' docs/deployment.md
test "$?" -eq 1
rg -q 'C .*mortred_ai_server_v1' docs/deployment.zh-cn.md
test "$?" -eq 1
```

期望两段均为退出码 **0**。

```bash
# A2：两份 deployment 的 :8080 用途列含字面 POST /v1/models/{id}/infer
# 且该行不得把未标明遗留的 /mortred_ai_server_v1 当作推荐推理入口
# 期望退出码 0
python3 - <<'PY'
from pathlib import Path
import sys
need = "POST /v1/models/{id}/infer"
legacy_ok = ("legacy", "遗留")
ok = True
for p in ["docs/deployment.md", "docs/deployment.zh-cn.md"]:
    hit = False
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if "| `8080`" not in line and "| 8080 |" not in line:
            continue
        if "mortred-gateway" not in line and "gateway" not in line.lower():
            continue
        hit = True
        if need not in line:
            print(f"FAIL {p}: :8080 row missing literal {need!r}\n  {line}")
            ok = False
        if "/mortred_ai_server_v1" in line and not any(t in line for t in legacy_ok):
            print(f"FAIL {p}: :8080 row treats unlabeled /mortred_ai_server_v1 as infer entry\n  {line}")
            ok = False
        else:
            print(f"OK {p}: :8080 purpose has {need!r} and legacy path (if any) is marked")
        break
    if not hit:
        print(f"FAIL {p}: no :8080 gateway purpose row")
        ok = False
sys.exit(0 if ok else 1)
PY
```

期望退出码 **0**。

```bash
# A5：两份 api-contract 分别含推荐路径字面量（不得一次 rg 扫两份）
rg -q -F 'POST /v1/models/{id}/infer' docs/api-contract.md
rg -q -F 'POST /v1/models/{id}/infer' docs/api-contract.zh-cn.md
```

期望两句均为退出码 **0**。

```bash
# A1+A5 对照：四份必读各自命中字面量（按文件分别 test，不得 files-with-matches 一次过四份）
rg -q -F 'POST /v1/models/{id}/infer' docs/api-contract.md
rg -q -F 'POST /v1/models/{id}/infer' docs/api-contract.zh-cn.md
rg -q -F 'POST /v1/models/{id}/infer' docs/deployment.md
rg -q -F 'POST /v1/models/{id}/infer' docs/deployment.zh-cn.md
```

期望四句均为退出码 **0**。

```bash
# A4：两份 api-contract 分别抽出 JSON 块再查四类键；OCR 只查键名，正文「禁止 text」不参与失败
# 期望退出码 0；缺一类、检测被 OCR 冒充、OCR 元素含键 text 则为 1
python3 - <<'PY'
from pathlib import Path
import json, re, sys

FENCE = re.compile(r"```(?:json)?\n(.*?)```", re.S)

def loads_all(text):
    out = []
    for m in FENCE.finditer(text):
        raw = m.group(1).strip()
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
    return out

def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for i in x:
            yield from walk(i)

def has_cls(objs):
    return any(isinstance(o, dict) and {"class_id", "category", "scores"} <= set(o) for o in objs)

def det_items(objs):
    items = []
    for o in objs:
        if isinstance(o, list):
            items.extend([i for i in o if isinstance(i, dict)])
        if isinstance(o, dict) and "data" in o and isinstance(o["data"], list):
            items.extend([i for i in o["data"] if isinstance(i, dict)])
    return items

def has_det(items):
    return any({"bbox", "detail_infos"} <= set(i) and "polygon" not in i for i in items)

def has_seg(objs):
    return any(isinstance(o, dict) and {"image", "colorized_mask"} <= set(o) for o in objs)

def ocr_blocks(items):
    return [i for i in items if isinstance(i, dict) and "polygon" in i]

ok = True
for p in ["docs/api-contract.md", "docs/api-contract.zh-cn.md"]:
    parsed = loads_all(Path(p).read_text(encoding="utf-8"))
    objs = list(walk(parsed))
    items = det_items(objs)
    ocr = ocr_blocks(items)
    fail = []
    if not has_cls(objs):
        fail.append("classification object missing class_id/category/scores")
    if not has_det(items):
        fail.append("detection array item missing bbox+detail_infos (or only OCR polygon items present)")
    if not has_seg(objs):
        fail.append("segmentation object missing image+colorized_mask")
    if not ocr:
        fail.append("OCR block missing polygon")
    elif any("text" in i for i in ocr):
        fail.append("OCR JSON element has key text")
    if fail:
        print(f"FAIL {p}: " + "; ".join(fail))
        ok = False
    else:
        print(f"OK {p}: four JSON shapes checked; OCR keyed without text")
sys.exit(0 if ok else 1)
PY
```

期望退出码 **0**。

```bash
# A3：四份按文件分别：出现 mortred_ai_server_v1 则同文件必须有 legacy 或 遗留
python3 - <<'PY'
from pathlib import Path
import sys
ok = True
files = [
    "docs/deployment.md", "docs/deployment.zh-cn.md",
    "docs/api-contract.md", "docs/api-contract.zh-cn.md",
]
for p in files:
    t = Path(p).read_text(encoding="utf-8")
    if "mortred_ai_server_v1" in t and ("legacy" not in t.lower() and "遗留" not in t):
        print(f"FAIL {p}: mortred_ai_server_v1 without legacy/遗留")
        ok = False
    else:
        print(f"OK {p}")
sys.exit(0 if ok else 1)
PY
```

期望退出码 **0**。四文件均无该路径时本命令仍为 **0**。

```bash
# A6：运行时与 OpenAPI 未改（无 diff 时期望退出码 0）
test -z "$(git diff --name-only -- src conf docs/openapi.json)"
```

期望退出码 **0**。

## RD 不得自行决定

- 不得把推荐入口改成 `{server_uri}`、模型直连端口或 `/v1/models/{id}/jobs`。
- 不得改 C++/gateway/OpenAPI 生成来「顺便统一路径」。
- 不得自造 `results[].data` 字段；OCR 不得加识别结果（JSON 示例元素不得出现键 `text`）。
- 不得只改英文或只改 `api-contract`；四份必读必须一起交。
- 不得用本条修 FAQ 结构、GPU pack、P1-3 安装或任何 P0。
- `{id}` 示例不得使用已废弃大写 HTTP 名。
- 不得恢复 A1「或等价」口径；架构图边标注必须含字面 `POST /v1/models/{id}/infer`。
- 不得用一次跨文件 `rg -q` 代替按文件自测；不得让 OCR JSON 的 `bbox`/`detail_infos` 充当检测示例。
