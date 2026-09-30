# 技术方案

- 问题: P1-3
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只收口遗留 `conf/trt_engines.json` 与 `conf/onnx_sources.json` 上 depth、metric3d、hrnet 相对产品真源的 engine 和 onnx 漂移，以及对齐检查未挂日常门禁。产品真源仍是 `conf/product_static_bs1.json` 与产品 toml。不把遗留清单升为产品入口。不改 convert 默认 manifest。不改进程模型、HTTP、后端 type。不改 yolov8s 路径。不扩大到 P1-4、D-2 或其他队列条。不对 onnx_sources 做全量 runtime_path 改写。

六对映射：

| 遗留 model | 产品 id |
| --- | --- |
| depth_anything_vits14 | depth_vits14 |
| depth_anything_vitb14 | depth_vitb14 |
| depth_anything_vitl14 | depth_vitl14 |
| metric3d_512x1088 | metric3d_512 |
| metric3d_1088x1920 | metric3d_1088 |
| hrnetw48_ccd_fp32 | hrnet |

产品 engine 为对应 `*.engine.static_bs1.fp16`，onnx 为 `*.static_bs1.onnx`。

## 做法

### 推荐

把这六行遗留 onnx/engine 改成与产品行一致。只把这六条 onnx_sources 的 runtime_path 改成产品 engine。扩展现有对齐脚本，用冻结映射比对，失败时点名产品路径和 `product_static_bs1`。在 `scripts/check_consistency.py` 里调用该检查，从而进入已有 CI。

### 否决

1. 只加门禁、不改遗留路径。convert 仍会建到裸 engine。
2. 把 convert 默认 manifest 改成产品文件。
3. 全量改写所有 onnx_sources runtime_path。

## 改动面

- `conf/trt_engines.json` 六行 onnx/engine。
- `conf/onnx_sources.json` 仅 HRNET、三条 DEPTH_ANYTHING、两条 METRIC3D 的 runtime_path。
- 对齐脚本与 `scripts/check_consistency.py` 的调用。

## 明确不改

产品 toml、`product_static_bs1.json`、convert 默认 MANIFEST、进程模型、HTTP、yolov8s 路径、doctor/init 起服语义、其他 onnx_sources 条目。

## 契约

HTTP 与 toml type 不变。失败文案必须含产品路径和 `product_static_bs1`。

## 性能与资源

检查只读 JSON，无 GPU、无 trtexec、不读权重是否存在。

## 验收标准

1. 六个遗留 model 的 onnx 与 engine 和产品对应 id 的字段完全一致。对齐检查退出码 0。
2. 与 hrnet、depth_vits14、depth_vitb14、depth_vitl14、metric3d_512、metric3d_1088 对应的六条 runtime_path 等于各自产品 engine。
3. 把任一映射行的遗留 engine 改回裸路径后，`python3 scripts/check_consistency.py` 退出码非 0，输出同时出现遗留路径、产品路径和 `product_static_bs1`。还原后再退出码 0。
4. `python3 scripts/check_consistency.py` 在对齐成立时退出码 0，并且会执行这六行的断言。
5. consistency 仍不要求全部产品 engine 都出现在遗留清单里。
6. yolov8s 仍对齐。convert 默认 MANIFEST 仍指向 `conf/trt_engines.json`。产品 toml 的 type 不改。

## 测试义务

QA 做一次「改坏 depth 或 hrnet engine → consistency 非 0 且点名产品路径 → 恢复再绿」。不测 GPU convert、起服、pack occupancy。

## 自测命令

```bash
python3 scripts/check_consistency.py
```

期望退出码 0。注入裸 engine 时期望非 0，还原后期望 0。

```bash
grep -n 'MANIFEST=.*trt_engines.json' scripts/convert_trt_engines.sh
```

期望退出码 0，且仍指向遗留文件。

## RD 不得自行决定

不得改 convert 默认 manifest。不得改产品 toml 去迁就裸路径。不得扩大映射表。门禁必须挂进 `check_consistency.py`。
