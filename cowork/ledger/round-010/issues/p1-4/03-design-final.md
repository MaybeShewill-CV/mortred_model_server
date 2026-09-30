# 技术方案

- 问题: P1-4
- 版本: v2
- 对应列表: cowork/ledger/round-010/02-p0-p1-final.md v2

## 边界

只收口两条可观察变化：交付用 pack 副本含本机校准结果；声明 `gpu_mem_mib` 与实测差超容差时有拒启文案。仓库里的 `conf/packs` 样例保持 seed。无 GPU 用夹具验收。不改进程模型。不新增 StatusCode。

本条的生产可观察入口是操作者提供的 `--measured-json`（catalog id → MiB），由 `scripts/pack_occupancy.py --check` 读入，并调用与 `occupancy_ready_for_spawn` 同一套容差和文案。supervisor 本条不新增 nvidia-smi 模型显存采样。未提供 measured-json 时，spawn 行为与现在一致。

## 做法

### 推荐

1. 用夹具对临时 pack 副本 `--write-pack`，使副本出现正的 `gpu_mem_mib` 且 source 不是 seed。直接写 `conf/packs` 必须失败且不改文件。
2. 提供了实测且 `|declared - measured| > max(64, declared/100)` 时失败。文案以 `occupancy gate:` 开头，含声明值、实测值和 `mortredctl calibrate --write-pack`。未提供实测时不因这条失败。
3. `pack_occupancy.py --check --measured-json` 是拒启文案的生产入口。supervisor 不新增模型显存采样。

### 否决

把本机指纹写进已入库 pack。把真 GPU calibrate 写成验收。在 supervisor 里用 nvidia-smi 采模型显存来代替 `--measured-json`。

## 改动面

- `src/control/occupancy_gate.h` 与 `test/occupancy_gate_unittest.cc`
- `scripts/pack_occupancy.py`
- 若 `ctest -R '^occupancy_gate_unittest$'` 选中 0 条，加 `add_test` 别名

## 明确不改

入库 pack 的 1800/seed、进程模型、HTTP、StatusCode、supervisor 的 nvidia-smi 采样面。

## 契约

未传 `--measured-json` 时行为与现在一致。超差文案须含声明值、实测值、`mortredctl calibrate` 和 `--write-pack`。

## 性能与资源

无额外常驻进程。不拉起模型。无实测时跳过比对。

## 验收标准

1. 临时 pack 经夹具 `--write-pack` 后有正的 `gpu_mem_mib` 且 source 不是 seed。`conf/packs/yolov8.toml` 仍是 1800 和 seed。
2. 对已入库 pack 直接 `--write-pack` 退出码非 0，文件不变。
3. 经 `--check --measured-json` 或单测，超差时失败串含声明值、实测值、`mortredctl calibrate` 和 `--write-pack`，并以 `occupancy gate:` 开头。无新 StatusCode。
4. 容差内通过。未提供实测时不因本条失败。
5. 以上在无 nvidia-smi 的夹具下通过。

## 测试义务

QA 覆盖临时写入、拒写入库、超差文案、容差内和无 measured-json。不测真 GPU。不要求 supervisor 新增显存采样。

## 自测命令

```bash
python3 scripts/calibrate_pack.py --self-test
python3 scripts/pack_occupancy.py --self-test
ctest --test-dir build/full -R '^occupancy_gate_unittest$' --output-on-failure
```

期望退出码 0。选中 0 条算失败。超差路径期望非 0，文案含三要素。

## RD 不得自行决定

不得把非 seed 写进 `conf/packs`。不得改 1800/seed。容差固定。未提供实测时不得让现有检查变红。不得用 nvidia-smi 采样代替 `--measured-json`。不得新增 StatusCode。

## 签字
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| cv-pm | 同意 | 无 | v2 | `cowork/ledger/round-010/issues/p1-4/03-discussion/v2-sign-pm.md` |
| cv-rd | 同意 | 无 | v2 | `cowork/ledger/round-010/issues/p1-4/03-discussion/v2-sign-rd.md` |
| cv-tech-lead | 同意 | 无 | v2 | `cowork/ledger/round-010/issues/p1-4/03-discussion/v2-sign-tech-lead.md` |
