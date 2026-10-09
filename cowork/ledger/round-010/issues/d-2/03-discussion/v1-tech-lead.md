# 技术方案

- 问题: D-2 attentive_gan.toml 输入尺寸仍 stale
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md 劣势 D-2

## 边界

上一版审查意见：无。

只收口 ATTENTIVE_GAN 去雨这一条已经冻结的变化：对外可读的 `model_input_image_size` 与已记录的 session 尺寸一致。键序是 `[height, width]`。

现在三份 toml 都写 `model_input_image_size = [240, 320]`：产品 `conf/model/enhancement/attentive_gan_derain/attentive_gan.toml`、`conf/ci/attentive_gan_onnx_hosted.toml`、`conf/bench/set_a/attentive_gan.toml`。后两份是产品 params 的副本，只改过 backend。Session 记录是高 240、宽 360：`conf/onnx_sources.json` 里 `ATTENTIVE_GAN_DERAIN` 的 `input_tensor` 形状为 `[1, 240, 360, 3]`，`scripts/export_onnx/export_attentive_gan.py` 为 `HEIGHT, WIDTH = 240, 360`。`src/models/enhancement/attentive_gan_derain_net.inl` 的 `on_init` 用 session `shape[1]`、`shape[2]` 作为高和宽，解码预算用的也是这个尺寸；`params` 只参与解码放大倍数。`contract.preprocess`、`onnx_source.notes`，以及 `scripts/export_onnx/update_onnx_sources_p2.py` 里的同句 notes，仍写着 toml `[240,320]` 且 stale。

产品 server 配置指向产品 toml，`server_uri` 为 `/mortred_ai_server_v1/enhancement/attentive_gan`。本条不改这条绑定。

不涉及 P0-1、P1-1 至 P1-6、D-1、D-3 至 D-6。

## 做法

### 推荐

把上述三份 toml 的 `model_input_image_size` 改成 `[240, 360]`。同时改掉仍声称 toml 为 `[240, 320]` 或尺寸 stale 的两处注记：`conf/onnx_sources.json` 中 `ATTENTIVE_GAN_DERAIN` 的 `contract.preprocess` 与 `onnx_source.notes`，以及 `scripts/export_onnx/update_onnx_sources_p2.py` 里同一条 notes。注记继续写明 session 为 NHWC `[1, 240, 360, 3]`、输出 HWC `[240, 360, 3]`、C++ 按 session 形状做预处理。

手改这几行。不要运行 `scripts/bench_set_a/generate.py`、`scripts/export_onnx/write_ci_overlays.py` 或 `scripts/export_onnx/update_onnx_sources_p2.py` 来刷文件。前两个会重写整批 overlay，第三个会对多个 id 做 `model.update`。

代价：三份 toml 各改尺寸一行，两份注记删掉「toml 仍是 320 / stale」这句。推理像素、解码预算、常驻进程和显存不变，因为 `on_init` 已经按 session 的 240×360 做 resize 和解码预算。不新建 engine。

### 否决

只在 `on_init` 里比对 toml 与 session，不一致就 `MODEL_INIT_FAILED` 并点名两边尺寸，toml 仍留 `[240, 320]`。产品配置今天能起来，因为初始化不读这个键。比对之后，已发布的产品 toml 会在启动失败。操作员打开 toml 仍然读到宽 320。冻结句里的「启动点名」以继续不一致为前提；本条要消掉的是已发布尺寸和 session 记录不一致。

把预处理改成按 toml 的 `[240, 320]` 送进网络，让运行时去迁就过期数字。导出脚本和 `input_tensor` 记录的静态形状是 `[1, 240, 360, 3]`。按 320 做 resize 会和这张静态图的宽不一致。

## 改动面

- `conf/model/enhancement/attentive_gan_derain/attentive_gan.toml`：`model_input_image_size`。
- `conf/ci/attentive_gan_onnx_hosted.toml`：同一键。
- `conf/bench/set_a/attentive_gan.toml`：同一键。
- `conf/onnx_sources.json`：仅 `id` 为 `ATTENTIVE_GAN_DERAIN` 的 `contract.preprocess` 与 `onnx_source.notes` 中声称 toml 为 320 或 stale 的句子。
- `scripts/export_onnx/update_onnx_sources_p2.py`：`ATTENTIVE_GAN_DERAIN` 那条 notes 里的同一句。

## 明确不改

- `src/models/enhancement/attentive_gan_derain_net.inl` 的 `on_init`、`preprocess`、`postprocess` 和状态码。resize 与解码预算继续用 session 的高和宽。
- `scripts/export_onnx/export_attentive_gan.py` 的 `HEIGHT, WIDTH = 240, 360`。
- 产品 toml 的 `type = "tensorrt"` 与 `attentive_gan_derain.engine.static_bs1.fp32`。
- CI toml 的 `type = "onnx"`、`device = "cpu"` 与 `attentive_gan_derain.static_bs1.onnx`。
- bench toml 的 `type = "tensorrt"` 与 `attentive_gan_derain.engine.static_bs1.fp32`。
- `conf/server/enhancement/attentive_gan_derain/attentive_gan_server_config.toml` 的 `server_uri` 与 `model_config_file_path`。
- `image_decode_mode`、`image_decode_budget_upscale`、`image_decode_backend`。
- `ATTENTIVE_GAN_DERAIN` 的 `product_type`、权重路径、license，以及其他 id。
- engine 文件、`mortredctl prepare`、gpu profile。
- 日常对齐门禁、doctor、init、CI。那是 P1-3。
- HTTP 路径、端口、worker、其他模型的 `model_input_image_size`、cowork 历史账本。

## 契约

`model_input_image_size` 仍是 `[height, width]`。三份 attentive_gan toml 的值改为 `[240, 360]`。

`ATTENTIVE_GAN_DERAIN.contract.inputs` 里 `input_tensor.shape` 保持 `[1, 240, 360, 3]`。`onnx_source.notes` 与生成脚本中的同句 notes 继续包含 `240x360` 和 `[1,240,360,3]`。`contract.preprocess`、这两处 notes 不再包含 `240,320`、`240, 320` 或 `stale`。

HTTP 路径、请求体和增强结果字段不变。tensor 布局保持静态 NHWC 输入、HWC 输出。不新增配置键，不新增状态码。

## 性能与资源

一个 catalog id 仍是一个 OS 进程。gateway 仍是 :8080，supervisor 仍是 :8787。本条不增加进程，不加载权重，不重建 engine，不改变显存占用。送入网络的高和宽、JPEG 解码预算继续是 session 的 240×360。cpu profile 不因此链接 TensorRT。

## 验收标准

1. 三份 toml（产品、`conf/ci/attentive_gan_onnx_hosted.toml`、`conf/bench/set_a/attentive_gan.toml`）的 `model_input_image_size` 均为 `[240, 360]`，且不再出现该键等于 `[240, 320]`。
2. 产品 toml 仍是 `type = "tensorrt"`，`model_file_path` 仍含 `attentive_gan_derain.engine.static_bs1.fp32`。CI toml 仍是 `type = "onnx"`、`device = "cpu"`，`model_file_path` 仍含 `attentive_gan_derain.static_bs1.onnx`。bench toml 仍是 `type = "tensorrt"`，`model_file_path` 仍含 `attentive_gan_derain.engine.static_bs1.fp32`。
3. server 配置的 `server_uri` 仍是 `/mortred_ai_server_v1/enhancement/attentive_gan`，`model_config_file_path` 仍是 `../conf/model/enhancement/attentive_gan_derain/attentive_gan.toml`。
4. `conf/onnx_sources.json` 中恰好一条 `id` 为 `ATTENTIVE_GAN_DERAIN`。其 `input_tensor.shape` 为 `[1, 240, 360, 3]`。该条的 `contract.preprocess` 与 `onnx_source.notes` 仍含 `240x360`，notes 仍含 `[1,240,360,3]`。这两段文字以及 `scripts/export_onnx/update_onnx_sources_p2.py` 全文都不再含 `240,320`、`240, 320` 或 `stale`。`scripts/export_onnx/export_attentive_gan.py` 仍含 `HEIGHT, WIDTH = 240, 360`。

## 测试义务

黑盒：在仓库根目录执行「自测命令」，退出码 0。该命令覆盖验收第 1 条至第 4 条。不启动 gateway 或 supervisor，不要求本机有权重或 GPU。

白盒：读 `attentive_gan_derain_net.inl` 的 `on_init` 与 `preprocess`。`_m_input_size_host` 的高、宽仍分别来自 session 输入 `shape[1]`、`shape[2]`。`resize` 与 `set_image_decode_hint` 仍使用这个 host 尺寸。本条没有把 `model_input_image_size` 改成 resize 的数据源。

## 自测命令

在仓库根目录执行。通过时期望退出码 0。任一断言失败时期望退出码 1。

```bash
python3 - <<'PY'
import json, pathlib, sys

root = pathlib.Path(".")
rels = [
    "conf/model/enhancement/attentive_gan_derain/attentive_gan.toml",
    "conf/ci/attentive_gan_onnx_hosted.toml",
    "conf/bench/set_a/attentive_gan.toml",
]
for rel in rels:
    text = (root / rel).read_text(encoding="utf-8")
    if "model_input_image_size = [240, 360]" not in text:
        print("size", rel)
        raise SystemExit(1)
    if "model_input_image_size = [240, 320]" in text:
        print("stale size", rel)
        raise SystemExit(1)

product = (root / rels[0]).read_text(encoding="utf-8")
ci = (root / rels[1]).read_text(encoding="utf-8")
bench = (root / rels[2]).read_text(encoding="utf-8")
if 'type = "tensorrt"' not in product or "attentive_gan_derain.engine.static_bs1.fp32" not in product:
    print("product backend")
    raise SystemExit(1)
if 'type = "onnx"' not in ci or 'device = "cpu"' not in ci or "attentive_gan_derain.static_bs1.onnx" not in ci:
    print("ci backend")
    raise SystemExit(1)
if 'type = "tensorrt"' not in bench or "attentive_gan_derain.engine.static_bs1.fp32" not in bench:
    print("bench backend")
    raise SystemExit(1)

server = (root / "conf/server/enhancement/attentive_gan_derain/attentive_gan_server_config.toml").read_text(encoding="utf-8")
if 'server_uri="/mortred_ai_server_v1/enhancement/attentive_gan"' not in server:
    print("uri")
    raise SystemExit(1)
if 'model_config_file_path="../conf/model/enhancement/attentive_gan_derain/attentive_gan.toml"' not in server:
    print("model path")
    raise SystemExit(1)

data = json.loads((root / "conf/onnx_sources.json").read_text(encoding="utf-8"))
rows = [m for m in data["models"] if m.get("id") == "ATTENTIVE_GAN_DERAIN"]
if len(rows) != 1:
    print("id count", len(rows))
    raise SystemExit(1)
row = rows[0]
shape = next(i.get("shape") for i in row["contract"]["inputs"] if i.get("name") == "input_tensor")
if shape != [1, 240, 360, 3]:
    print("shape", shape)
    raise SystemExit(1)
claim = row["contract"].get("preprocess", "") + "\n" + row.get("onnx_source", {}).get("notes", "")
gen = (root / "scripts/export_onnx/update_onnx_sources_p2.py").read_text(encoding="utf-8")
export = (root / "scripts/export_onnx/export_attentive_gan.py").read_text(encoding="utf-8")
if "240x360" not in claim or "[1,240,360,3]" not in row["onnx_source"]["notes"]:
    print("session fact missing")
    raise SystemExit(1)
if "240x360" not in gen or "HEIGHT, WIDTH = 240, 360" not in export:
    print("generator fact missing")
    raise SystemExit(1)
for token in ("240,320", "240, 320", "stale"):
    if token in claim or token in gen:
        print("claim", token)
        raise SystemExit(1)
print("ok")
PY
```

## RD 不得自行决定

尺寸只能是 `[240, 360]`，即高 240、宽 360。不得写成 `[360, 240]` 或其他数字。

不得用启动失败代替改数字。不得让 `on_init` 改读 toml 尺寸来做 resize。本机若加载到的静态输入形状不是 `[1, 240, 360, 3]`，停下来报告，不要改图，也不要把 toml 改去迁就另一张图。

不得重跑 Set A 生成、CI overlay 重写或 `update_onnx_sources_p2.py`。不得改 backend 类型、权重路径、`server_uri` 或 `model_config_file_path`。不得重新导出 ONNX，不得 `mortredctl prepare`。
