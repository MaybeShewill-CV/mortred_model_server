# 技术方案

- 问题: P1-4
- 版本: v1
- 对应列表: `cowork/ledger/round-002/02-p0-p1-final.md` P1-4

## 边界

- 只在 `docs/api-contract.md` 与 `docs/api-contract.zh-cn.md` 追加抠图、增强、深度的 `results[].data` 样例。
- OpenAPI：这三族必填只有 `image`。不得给抠图加 `colorized_mask`。分割样例保持 `image` + `colorized_mask`。
- 既有分类、检测、分割、OCR 四段不得改坏。catalog id 用 `modnet`、`realesrgan`、`metric3d_512`。
- 三份 JSON 键集合都是 `{image}`，族别靠紧邻标题区分。中英都要有。
- 不改 `src/`、`conf/`、`docs/openapi.json`。不启服务。测试在 WSL 用 python3。
- 不要把这三个 id 写进样例节之前的那一行举例列表，否则自测会把后面已有的 JSON 误当成这三族。

## 做法

### 推荐

在 OCR 样例之后追加三段。英文标题含 `` `modnet` ``、`` `realesrgan` ``、`` `metric3d_512` ``。中文标题同样含这三个反引号 id。每段后的第一个 ```json 对象只有键 `image`。

### 否决

改 OpenAPI 或运行时。三族合成一段共享 JSON。只改一种语言。

## 改动面

- `docs/api-contract.md`
- `docs/api-contract.zh-cn.md`

## 明确不改

`src/`、`conf/`、`docs/openapi.json`、既有四段 JSON、进程与推理路径。

## 验收标准

- **A1** 英文契约在按任务节内，紧邻标题分别含 `` `modnet` ``、`` `realesrgan` ``、`` `metric3d_512` ``；每段后第一个 JSON 对象的键集合正好是 `{image}`，不含 `colorized_mask`。
- **A2** 中文契约同样满足。
- **A3** 既有四段仍在：分类含 `class_id`/`category`/`scores`；检测含 `bbox`+`detail_infos`；分割含 `image` 与 `colorized_mask`；OCR 数组项含 `polygon` 且无键 `text`。
- **A4** 紧邻 `` `modnet` `` 的 JSON 无 `colorized_mask`。
- **A5** `src/`、`conf/`、`docs/openapi.json` 无改动。

## 自测命令

在仓库根、WSL 执行。全部通过时期望退出码 0。

```bash
python3 - <<'PY'
from pathlib import Path
import json, re, sys

def extract_after_id(text, catalog_id):
    pat = re.compile(
        rf"(?m)^.*`{re.escape(catalog_id)}`.*\n+(?:.*\n)*?```json\n(.*?)\n```",
        re.S,
    )
    m = pat.search(text)
    if not m:
        raise AssertionError(f"missing block for {catalog_id}")
    return json.loads(m.group(1))

def check_file(path):
    text = Path(path).read_text(encoding="utf-8")
    for cid in ("modnet", "realesrgan", "metric3d_512"):
        obj = extract_after_id(text, cid)
        assert isinstance(obj, dict), cid
        assert set(obj.keys()) == {"image"}, (cid, set(obj.keys()))
        assert "colorized_mask" not in obj, cid
    assert "`mobilenetv2`" in text and "class_id" in text
    assert "`yolov8s`" in text and "detail_infos" in text
    seg = extract_after_id(text, "pphuman_server")
    assert {"image", "colorized_mask"} <= set(seg.keys())
    ocr = extract_after_id(text, "dbnet")
    assert isinstance(ocr, list) and ocr and "polygon" in ocr[0] and "text" not in ocr[0]

ok = True
for p in ("docs/api-contract.md", "docs/api-contract.zh-cn.md"):
    try:
        check_file(p)
        print("PASS", p)
    except Exception as e:
        ok = False
        print("FAIL", p, e)
sys.exit(0 if ok else 1)
PY
```

## RD 不得自行决定

不得给这三族加 `colorized_mask`。不得合并成一段。不得改既有四段。不得改 openapi/src/conf。不得只改一种语言。id 必须是 `modnet`、`realesrgan`、`metric3d_512`。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v1 | `cowork/ledger/round-002/issues/p1-4/03-discussion/v1-sign-pm.md` |
| cv-rd | 同意 | 无 | v1 | `cowork/ledger/round-002/issues/p1-4/03-discussion/v1-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v1 | `cowork/ledger/round-002/issues/p1-4/03-discussion/v1-sign-tech-lead.md` |
