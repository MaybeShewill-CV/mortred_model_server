# 技术方案

- 问题: P1-2
- 版本: v2
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md`（列表版本 v2）

## 边界

代表性 pack 带可复用 occupancy 样本；换机可复跑 calibrate、写回、doctor。不改 P0-1 fail-closed，不做 P1-3、P1-4。demo 保持 MNN cpu、不要求 stamp。`--write-pack` 仍拒绝直接写 `conf/packs/`。Git 内 MiB 是 seed，不是某台卡的瞬时 nvidia-smi。规则 20 只按改动面放宽，不得整条删掉。

## 做法

### 推荐

在 `conf/packs/yolov8.toml`、`conf/packs/metric3d.toml` 写入正的 `gpu_mem_mib`、与 `worker_nums` 一致的 `gpu_mem_at_workers`、`gpu_mem_source = "seed"`。不写 `gpu_name` / `gpu_memory_total_mib`。

新增 `conf/packs/gpu_yolo_metric3d.toml`，含 `yolov8s` 与 `metric3d_512`，同样的 seed 字段。注释写明必须被本机 `calibrate --write-pack` 覆盖，并给出换机命令。

`scripts/check_consistency.py` 规则 20：仅当 `gpu_mem_source="seed"`、正的 `gpu_mem_mib`、`gpu_mem_at_workers` 与 `worker_nums` 一致、且没有 `gpu_name` / `gpu_memory_total_mib` 时允许提交。其它 `gpu_mem_mib` 仍禁止。

`docs/deployment.md` 与中文对等写成一条链：copy pack、prepare、calibrate `--write-pack`、doctor、start。

换机命令：

```bash
cp conf/packs/gpu_yolo_metric3d.toml /tmp/machine-pack.toml
mortredctl prepare --pack /tmp/machine-pack.toml
mortredctl calibrate --pack /tmp/machine-pack.toml --write-pack
export MORTRED_PACK=/tmp/machine-pack.toml
mortredctl doctor
```

### 否决

只改文档不写 stamp；放开直接写 git pack；只在旁路放样本；整条删除规则 20。

## 改动面

- `conf/packs/yolov8.toml`、`conf/packs/metric3d.toml`、新建 `conf/packs/gpu_yolo_metric3d.toml`
- `docs/deployment.md`、`docs/deployment.zh-cn.md`
- `scripts/check_consistency.py` 规则 20 的 seed 例外
- 需要时只扩展 self-test，不改门禁语义

## 明确不改

demo.toml、对 `conf/packs/` 的拒写、P0-1、P1-3、P1-4、产品 toml。

## 契约

seed 段可进 git。非 seed 或带本机指纹的 `gpu_mem_mib` 不得合入。

## 性能与资源

不改变一 catalog 一进程。seed 必须保守，换机必须重校准。

## 验收标准

1. **AC1**：demo `--check` 与 `--self-test` 均为 0。
2. **AC2**：yolov8 与 metric3d `--check` 为 0，含 seed 三字段，无 `gpu_name`。
3. **AC3**：多 id pack `--check` 为 0，每个 TRT id 有正的 `gpu_mem_mib`。
4. **AC4**：临时拷贝去掉 `gpu_mem_mib` 后 `--check` 非 0，doctor 对该临时 pack 的 occupancy 门禁也非成功。
5. **AC5**：直接对 git pack `--write-pack` 非 0；对 `/tmp` 副本成功。
6. **AC6**：中英 deployment 能找到 copy、prepare、calibrate `--write-pack`、doctor、start，以及 seed 可被覆盖。
7. **AC7**：有 GPU 时换机链跑通后 `--check` 为 0。无 GPU 记 blocked，不阻碍 AC1–AC6 与 AC8。
8. **AC8**：规则 20 不对三个 seed pack 报 machine-local / occupancy stamp。临时拷贝去掉 `gpu_mem_source`、改成非 seed、或写入 `gpu_name` / `gpu_memory_total_mib` 时必须报错。不以整份 `check_consistency.py` 退出码 0 为验收。既有 openapi/catalog 红项不在本条。

## 测试义务

QA 执行 AC1–AC6 与 AC8。有 GPU 再补 AC7。

## 自测命令

| 命令 | 期望退出码 |
| --- | --- |
| `python3 scripts/pack_occupancy.py --self-test` | `0` |
| `python3 scripts/calibrate_pack.py --self-test` | `0` |
| 三个 seed pack 与 demo 的 `--check` | `0` |
| 规则 20 对三个 seed pack | 无 machine-local 报错 |
| 临时拷贝去掉 `gpu_mem_source` 或写入 `gpu_name` 后规则 20 | `≠0` |
| 去掉 `gpu_mem_mib` 后 `--check` | `≠0` |
| 对 git pack `--write-pack` | `≠0` |
| 对 `/tmp` 副本 `--write-pack --from-json` | `0` |
| doctor 指向剥 stamp 临时 pack | `≠0` |

## RD 不得自行决定

不得把 seed 写成换机无需重跑。不得写入 `gpu_name`。不得改 demo。不得 fail-open。不得放开 git `--write-pack`。不得删除规则 20。seed MiB 必须写明参考来源。
