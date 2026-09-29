# 技术方案

- 问题: P1-2
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md`（列表版本 v2）

## 边界

- 只落地冻结原意：代表性 pack 带可复用 occupancy 样本；换机可复跑 `calibrate → 写回 → doctor/spawn` 一条链。
- 不改 P0-1 的 doctor/pack occupancy fail-closed；不碰 P1-3、P1-4。
- `conf/packs/demo.toml` 保持 MNN cpu，不要求 occupancy stamp。`python3 scripts/pack_occupancy.py --self-test` 必须继续断言 demo 不触发 stamp。
- `calibrate_pack.py` 继续拒绝直接 `--write-pack` 写入 `conf/packs/`。写回只允许机器本地 pack 副本。
- Git 内 MiB 是 seed 起始样本，不是某一台卡瞬时 nvidia-smi 的唯一真理。换机必须用文档里的重跑命令覆盖。

## 做法

### 推荐

在 `conf/packs/yolov8.toml`、`conf/packs/metric3d.toml` 为 TensorRT id 写入 `gpu_mem_mib`（>0）、`gpu_mem_at_workers`（与 `worker_nums` 一致）、`gpu_mem_source = "seed"`。不写 `[pack]` 的 `gpu_name` / `gpu_memory_total_mib`。

新增 `conf/packs/gpu_yolo_metric3d.toml`，至少含 `yolov8s` 与 `metric3d_512` 两个 TensorRT catalog，各自同样的 seed 字段。

每个 seed 段用注释写明：数值是起始样本，必须被本机 `calibrate --write-pack` 覆盖，并给出换机命令。

`docs/deployment.md` 与中文对等把校准收成一条链：`copy pack → prepare → calibrate --write-pack → doctor → start`。允许 `gpu_mem_source=seed` 的起始样本，禁止把本机指纹或实测结果当 git 真源。

缺 stamp 的负例用临时拷贝剥掉 stamp，不改 demo。

换机命令：

```bash
cp conf/packs/gpu_yolo_metric3d.toml /tmp/machine-pack.toml
mortredctl prepare --pack /tmp/machine-pack.toml
mortredctl calibrate --pack /tmp/machine-pack.toml --write-pack
export MORTRED_PACK=/tmp/machine-pack.toml
mortredctl doctor
```

### 否决

1. 只改文档、不提交任何 `gpu_mem_mib`。不满足「代表性 pack 带可复用 occupancy」。
2. 放开 `--write-pack` 直接改 `conf/packs/`，并把某次本机读数当 git 唯一真源。
3. 只在旁路 examples 放样本，主示例 `yolov8`/`metric3d` 仍无 stamp。

## 改动面

- `conf/packs/yolov8.toml`、`conf/packs/metric3d.toml`
- 新增 `conf/packs/gpu_yolo_metric3d.toml`
- `docs/deployment.md`、`docs/deployment.zh-cn.md`
- 若 seed 字段需要断言，只扩展 `pack_occupancy.py` / `calibrate_pack.py` 的 self-test，不改门禁语义。

## 明确不改

demo.toml、对 `conf/packs/` 的拒写、P0-1 fail-closed、P1-3、P1-4、产品 toml 与 engine 真源。

## 契约

代表性 git pack 的每个 TRT id：`gpu_mem_mib` 为正整数，`gpu_mem_at_workers` 等于 `worker_nums`，`gpu_mem_source = "seed"`。不提交 `gpu_name` / `gpu_memory_total_mib`。`gpu_mem_source="seed"` 表示可被本机校准覆盖的起始样本。

## 性能与资源

不改变一 catalog 一进程。seed 过小会误放行，过大误拒，所以必须保守，换机必须重校准。engine 仍只由本机 `prepare` 生成。

## 验收标准

1. **AC1**：`pack_occupancy.py --check` 对 `conf/packs/demo.toml` 退出码 0，且 `--self-test` 退出码 0。
2. **AC2**：`yolov8.toml` 与 `metric3d.toml` 的 `--check` 均为 0；文件含 `gpu_mem_mib`、`gpu_mem_at_workers`、`gpu_mem_source = "seed"`，且没有把本机 `gpu_name` 写成必填真源。
3. **AC3**：新增多 id pack 至少两个 TensorRT catalog，`--check` 为 0，每个 TRT id 都有正的 `gpu_mem_mib`，且 `gpu_mem_at_workers` 与 `worker_nums` 一致。
4. **AC4**：拷贝代表性 TRT pack 到临时路径并删除 `gpu_mem_mib` 后，`--check` 退出码非 0，输出含 `gpu_mem_mib` 或 `calibrate`。`MORTRED_PACK` 指向该临时 pack 时，无 `--strict` 的 `mortredctl doctor` 对 occupancy 门禁也非成功退出。
5. **AC5**：对 `conf/packs/yolov8.toml` 直接 `--write-pack` 失败（非 0）。对 `/tmp` 副本 `--write-pack` 成功。
6. **AC6**：`docs/deployment.md` 与中文对等能找到 copy → prepare → calibrate `--write-pack` → doctor → start，以及 seed 可被覆盖的说明。
7. **AC7**：有 GPU 时按换机命令对机器本地 pack 跑通后 `--check` 为 0。无 GPU 时本条记 blocked，不阻碍 AC1–AC6 合入。

## 测试义务

QA 执行 AC1–AC6。有 GPU 再补 AC7。负例必须用临时剥 stamp pack。`pack_occupancy.py --self-test` 与 `calibrate_pack.py --self-test` 均为 0。

## 自测命令

| 命令 | 期望退出码 |
| --- | --- |
| `python3 scripts/pack_occupancy.py --self-test` | `0` |
| `python3 scripts/calibrate_pack.py --self-test` | `0` |
| `python3 scripts/pack_occupancy.py --project-root . --pack conf/packs/demo.toml --check` | `0` |
| `python3 scripts/pack_occupancy.py --project-root . --pack conf/packs/yolov8.toml --check` | `0` |
| `python3 scripts/pack_occupancy.py --project-root . --pack conf/packs/metric3d.toml --check` | `0` |
| `python3 scripts/pack_occupancy.py --project-root . --pack conf/packs/gpu_yolo_metric3d.toml --check` | `0` |
| 拷贝 yolov8 pack 并去掉 `gpu_mem_mib` 后 `--check` | `≠0` |
| `mortredctl calibrate --pack conf/packs/yolov8.toml --write-pack` | `≠0` |
| 对 `/tmp` 副本 `--write-pack --from-json` | `0` |
| `MORTRED_PACK` 指向剥 stamp 临时 pack 的 `mortredctl doctor` | `≠0` |

## RD 不得自行决定

不得把 seed 写成「已在本仓库 GPU 上校准完成、换机无需重跑」。不得在 git seed 写入 `gpu_name` 或 `gpu_memory_total_mib`。不得修改 demo 为需要 stamp。不得把 doctor 改回 fail-open。不得放开对 `conf/packs/` 的 `--write-pack`。不得做 P1-3 或 P1-4。seed 的 MiB 必须在实现说明里写明参考来源。
