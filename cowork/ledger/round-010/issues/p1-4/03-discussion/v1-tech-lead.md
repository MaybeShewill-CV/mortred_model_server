# 技术方案

- 问题: P1-4
- 版本: v1
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只收口两条可观察变化：交付用 pack 副本含本机校准结果；声明 `gpu_mem_mib` 与实测差超容差时有拒启文案。仓库里的 `conf/packs` 样例保持 seed，含 `yolov8.toml` 的 1800。不得把本机 gpu_name、总显存或非 seed 的 source 写进已入库 pack。无 GPU 用夹具验收。不改进程模型。不新增 StatusCode。不扩大到其他队列条。

## 做法

### 推荐

1. 用夹具 JSON 对临时 pack 副本执行 calibrate `--write-pack`，使副本出现正的 `gpu_mem_mib` 且 `gpu_mem_source` 不是 seed。入库 seed 不变。直接对 `conf/packs` `--write-pack` 必须失败且不改文件。
2. `occupancy_ready_for_spawn` 在提供了实测占用、且 `|declared - measured| > max(64, declared/100)` 时失败。文案走现有 `occupancy gate:`，同时含声明值、实测值和 `mortredctl calibrate --write-pack`。未提供实测时不因这条新检查失败。
3. 单测和 `pack_occupancy.py --self-test` 覆盖超差、容差内、无实测。全程不依赖 nvidia-smi。

### 否决

把本机指纹写进已入库 pack。把真 GPU calibrate 写成验收。新增 StatusCode。只改文档。

## 改动面

- `src/control/occupancy_gate.h` 与 `test/occupancy_gate_unittest.cc`
- `scripts/pack_occupancy.py`，必要时微调 `scripts/calibrate_pack.py` 自测

## 明确不改

入库 pack 的 seed 数字、进程模型、HTTP、StatusCode、其他队列条。

## 契约

拒启串前缀仍是 `occupancy gate:`。未提供实测时行为与现在一致。

## 性能与资源

无额外常驻进程。不拉起模型。无实测时跳过比对。

## 验收标准

1. 把 seed pack 拷到临时路径，用夹具 `--from-json --write-pack` 后，临时文件有正的 `gpu_mem_mib` 且 `gpu_mem_source` 不是 seed。`conf/packs/yolov8.toml` 仍是 1800 和 seed。
2. 对已入库 `conf/packs/*.toml` 直接 `--write-pack` 退出码非 0，且文件内容不变。
3. 声明与实测差大于 `max(64, declared/100)` 时，失败串同时含声明值、实测值、`mortredctl calibrate` 和 `--write-pack`，并且以 `occupancy gate:` 开头。无新 StatusCode。
4. 差在容差内时通过。未提供实测时不因本条失败。
5. 验收 1–4 在无 nvidia-smi 的夹具下通过。

## 测试义务

QA 跑临时 pack 写入、拒写入库样例、超差文案和容差内通过。确认 yolov8.toml 的 seed 未改。不测真 GPU calibrate。

## 自测命令

```bash
python3 scripts/calibrate_pack.py --self-test
python3 scripts/pack_occupancy.py --self-test
```

期望退出码 0。occupancy_gate 单测期望退出码 0。超差检查路径期望非 0，且文案含三要素。

## RD 不得自行决定

不得把非 seed 写进 `conf/packs`。不得改 1800/seed 冒充已校准。容差固定为 `max(64, declared/100)`。未提供实测时不得让现有检查变红。
