# 技术方案

- 问题: P2-9
- 版本: v1
- 对应列表: `cowork/ledger/round-001/02-p0-p1-final.md` v3 的 P2-9

## 边界

第 1 轮方案 v1。上一版审查意见无。

本方案只让这一件事成立：在仓库根执行 `python3 scripts/golden_drift_check.py --check`，退出码为 0；`test/golden_baseline.json` 的 `case_names` 含 `dinov2_vits14` 与 `metric3d_512`，`golden_sha256` 含键 `dinov2_vits14.json` 与 `metric3d_512.json`。

P2-10、P2-11 以及其他 id 不在本方案内。不跑模型推理，不启动 gateway、supervisor 或模型进程。GPU、engine、HTTP 吞吐不作为本条通过。

核对 `scripts/golden_drift_check.py` 与 `test/golden_baseline.json` 之后，这条检查的比较方式如下。

`case_names` 整表按顺序相等。脚本用正则 `(?:TEST\(model_golden,|GOLDEN_\w+_CASE\()\s*(\w+)` 只扫描 `test/model_golden_test.cc` 的源码文本，按出现顺序收集名字。当前源码得到 49 个名字。baseline 现有 30 个，这 30 个都还在源码里，相对顺序也还在。缺的是插在 `superpoint_feature_point` 与 `fastsam_segmentation` 之间的 19 个：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。多出来的名字会打印 `cases added` 并返回 1。两边名字相同而顺序不同时，打印 `case declaration order changed` 并返回 1。

`golden_sha256` 是 `test/golden/` 顶层每个文件的原始字节 SHA-256。多一个文件打印 `golden file added`，少一个打印 `golden file removed`，字节不同打印 `golden file modified`，然后返回 1。比较对象是文件字节，脚本不读取 json 内部的数值字段。

`case_count` 与 `golden_count` 由 `--record` 写成名单长度和文件个数。`--check` 不单独比较这两项。

`TRT_PRODUCT_GOLDEN(...)` 与上述正则对不上，这些测试名留在宏调用里，不进入 `case_names`。同目录的 `trt_*.json` 仍是文件，会计入字节哈希。当前有 18 个 `trt_*.json`，baseline 里没有这些键。

baseline 现有 26 个哈希键：20 个 json 与 6 个 png。这 20 个 json 仍在目录里。6 个 png（`attentivegan_enhancement.png`、`enlightengan_enhancement.png`、`fastsam_segmentation.png`、`realesrgan_enhancement.png`、`sam_automask_generation.png`、`sam_prompt_prediction.png`）也仍在。目录里另有 36 个 json 不在 baseline 中，其中包括 `dinov2_vits14.json`、`metric3d_512.json` 和 18 个 `trt_*.json`。`msocrnet` 有用例名，目录里没有 `msocrnet.json`。`metric3d_512.json` 约 5MB，内容是已提交的期望数值。

因此，退出码 0 要求 baseline 的整份 `case_names` 与上述 49 个名字逐项同序，并且 `golden_sha256` 覆盖目录顶层的每一个当前文件，每个值等于该文件当前字节的 SHA-256。锁定句点名的两个名字和两个 json 键是这张整表里的必含项。只写入这两项时，其余 17 个名字和其余新增文件仍会让退出码保持 1。

已在 baseline 里、字节未变的文件，重新计算得到的十六进制与现字符串相同，期望数值保持原文件。本步在只读限制下没有执行哈希进程，所以不宣称那 26 个已记录文件的字节已经变过。若记账前的 `--check` 打印 `golden file modified`，退出码 0 要求把该键改成当前文件字节的 SHA-256，文件本身保持不动。目录里的 json 和 png 已经在，这条检查按这些字节记账。

## 做法

### 推荐

只更新 `test/golden_baseline.json`。在 WSL 仓库根 `/mnt/g/Codex/mortred_model_server` 先执行一次 `python3 scripts/golden_drift_check.py --check`，阅读 drift 行，再执行同脚本的 `--record`，最后再执行 `--check`。

`--record` 调用脚本里现成的 `collect()`：按源码顺序写下 49 个 `case_names`（含 `dinov2_vits14` 与 `metric3d_512`），对 `test/golden/` 顶层每个文件做字节 SHA-256（含 `dinov2_vits14.json`、`metric3d_512.json`、其余新增 json、18 个 `trt_*.json`，以及仍在目录里的 6 个 png），并把 `case_count`、`golden_count` 写成这两个长度。写入格式沿用脚本现有的 `indent=2`、`sort_keys=True` 和末尾换行。`sort_keys` 只排列对象键；`case_names` 数组保持源码顺序。

记账前的 `--check` 若只有 `cases added` 与 `golden file added`，直接 `--record`。若同时有 `golden file modified`，仍然 `--record`，让该键改为当前字节的哈希，对应文件保持原字节。若出现 `cases removed`、`golden file removed` 或 `case declaration order changed`，停止。按本步核对，后三类不应出现：30 个旧名字都还在，20 个旧 json 和 6 个 png 都还在。

代价：baseline 从 30 个名字、26 个哈希，变成当前源码的 49 个名字和当前目录顶层文件的哈希。按现有文件是 56 个 json 加 6 个 png，共 62。此后这些文件的任一字节变化都会让 `--check` 返回 1。服务进程、显存、engine、HTTP 契约保持现状。`metric3d_512.json` 保持约 5MB 的原字节，baseline 里只增加它的一行哈希。

### 否决

只把 `dinov2_vits14`、`metric3d_512` 写进 `case_names`，并只补 `dinov2_vits14.json`、`metric3d_512.json` 两个哈希。代价是 diff 小。其余 17 个源码名字和其余新增文件（含 18 个 `trt_*.json`）仍会打印 added，`--check` 退出码仍为 1。锁定的通过包含退出码 0。不采用。

改 `golden_drift_check.py`，使两个名字和两个键存在就返回 0；或者从 `model_golden_test.cc` 删掉多出的用例、从 `test/golden/` 删掉多出的 json，让旧的 30/26 重新对上；或者用 `MORTRED_UPDATE_GOLDEN=1` 跑 `model_golden_test` 重写期望数值后再记账。收窄脚本之后，`check_consistency.py` 所调用的 `--check` 不再核对整份名单和每个文件的字节。删用例或删 json 会改掉现有测试期望；删掉 `dinov2_vits14` 与 `metric3d_512` 之后，锁定句要求的两个名字不再存在。重写 golden 会改动已提交 json 和 png 里的期望数值，并且要把 GPU 推理当成这条检查的通过手段。脚本比的是已有文件的字节哈希，目录里的文件已经在。不采用。

## 改动面

- `test/golden_baseline.json`：整份换成当前 `collect()` 的结果。`case_names` 为源码中的 49 个名字，顺序与声明顺序相同。`golden_sha256` 为目录顶层每个文件名到其当前字节 SHA-256 的映射。`case_count` 为 49。`golden_count` 为顶层文件数，按现有目录为 62。

## 明确不改

- `scripts/golden_drift_check.py` 的正则、字节哈希和退出码。
- `scripts/check_consistency.py` 与 `.github/workflows/ci.yml`。`check_golden_zero_drift()` 继续调用同一条 `--check`。
- `test/model_golden_test.cc`、`test/model_golden_registry.h`。`dinov2_vits14` 与 `metric3d_512` 的 `GOLDEN_*_CASE` 声明保持原样。`msocrnet` 的已有声明留在源码里；本条不运行该模型，也不为它新建 json。`TRT_PRODUCT_GOLDEN` 保持原样，其测试名不写入 `case_names`。
- `test/golden/` 下全部文件的字节，包括 `dinov2_vits14.json`、`metric3d_512.json`、18 个 `trt_*.json` 和 6 个 png。
- 产品 toml、server toml、gateway、supervisor、TensorRT execution context、engine、HTTP 与 tensor 契约。
- `docs/ci-golden-regression.md` 里已有的 `--record` 然后 `--check` 说明。本条按这段现有步骤记账，文档保持不动。

## 契约

HTTP 路径、请求体、响应字段和 tensor 形状保持现状。

会变的是 `test/golden_baseline.json` 这一份检查账本：

- `case_names` 从 30 项变为与源码声明顺序一致的 49 项，其中含 `dinov2_vits14` 与 `metric3d_512`。
- `golden_sha256` 的键覆盖 `test/golden/` 顶层全部当前文件。值是文件字节的 SHA-256 小写十六进制，其中含 `dinov2_vits14.json` 与 `metric3d_512.json`。
- 对象键经 `--record` 排序。`case_names` 的数组顺序保持源码顺序。

`case_count`、`golden_count` 随这两份实际长度更新。`--check` 的通过取决于名单顺序和字节哈希。

## 性能与资源

本条不拉起常驻进程，不占用模型显存，不生成或加载 engine。每个 worker 的 TensorRT execution context 归属保持现状。检查只读 `test/model_golden_test.cc` 和 `test/golden/` 顶层文件并计算 SHA-256。`metric3d_512.json` 约 5MB，哈希一次即可，该文件不进入推理会话。

## 验收标准

1. 在 `/mnt/g/Codex/mortred_model_server` 执行 `python3 scripts/golden_drift_check.py --check`，进程退出码为 0，标准输出包含 `zero drift:`。
2. 打开 `test/golden_baseline.json`，`case_names` 含 `dinov2_vits14` 与 `metric3d_512`。
3. 同一文件的 `golden_sha256` 含键 `dinov2_vits14.json` 与 `metric3d_512.json`。
4. 第 3 条两个键的值等于当时这两个文件字节的 SHA-256。第 1 条退出码为 0 时，这一点与目录内其余文件的字节哈希同时成立。任一文件字节与账本不一致时，该命令退出码为 1。

## 测试义务

QA 做白盒，不跑 GPU，不跑 `model_golden_test`。推理成功不作为本条通过。

- 看本条 diff：产品文件只有 `test/golden_baseline.json`。`test/model_golden_test.cc` 与 `test/golden/` 下的文件不在 diff 里。
- 源码里仍能看到 `GOLDEN_FEATURE_EMBEDDING_CASE(dinov2_vits14` 与 `GOLDEN_DEPTH_CASE(metric3d_512`。
- `case_names` 里不出现 `TRT_PRODUCT_GOLDEN` 那一组测试名，例如名单中不出现 `trt_dinov2_vits14`。`golden_sha256` 中有 `trt_dinov2_vits14.json` 与 `trt_metric3d_512.json` 这类文件键。第 1 条退出码为 0 时，整表已经与 `collect()` 一致；QA 用这两个键确认文件哈希和用例名没有被混写。
- 执行「自测命令」的两条命令，退出码都是 0。
- `python3 scripts/check_consistency.py` 的整份退出码不作为本条通过。该脚本的 `check_golden_zero_drift()` 会执行同一条 `--check`；脚本里其余检查属于其他条目。

## 自测命令

在 WSL Ubuntu 中执行。两条都期望退出码 0。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 scripts/golden_drift_check.py --check
```

期望退出码：0

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import json, sys
b = json.load(open("test/golden_baseline.json", encoding="utf-8"))
ok = (
 "dinov2_vits14" in b["case_names"]
 and "metric3d_512" in b["case_names"]
 and "dinov2_vits14.json" in b["golden_sha256"]
 and "metric3d_512.json" in b["golden_sha256"]
)
sys.exit(0 if ok else 1)
PY
```

期望退出码：0

## RD 不得自行决定

- 不得修改 `test/golden/` 下任何文件的字节，也不得用 `MORTRED_UPDATE_GOLDEN` 或 GPU 推理重写期望数值。
- 不得删除、改名或调整 `test/model_golden_test.cc` 里的用例来换取退出码 0。不得为 `msocrnet` 新建 golden 文件。
- 不得把 `TRT_PRODUCT_GOLDEN` 的测试名写入 `case_names`，也不得从 `golden_sha256` 里去掉目录中仍存在的 `trt_*.json`。
- 不得修改 `golden_drift_check.py` 的比较范围，使「只含两个名字和两个键」返回 0。
- 不得只补 `dinov2_vits14` 与 `metric3d_512` 这两项。整表必须与当前 `collect()` 一致，名单顺序与源码声明顺序一致。
- 记账前若看到 `golden file modified`，只把 baseline 中该键更新为当前字节的 SHA-256，文件保持原字节。若看到 `cases removed`、`golden file removed` 或 `case declaration order changed`，停止并保持 golden 文件不动，不靠删除账本键换取退出码。
- 不得改 gateway、supervisor、产品 toml、engine，不得把本条扩到 P2-10、P2-11 或 `check_consistency.py` 的其他检查。
- `case_count` 与 `golden_count` 使用 `--record` 写出的实际长度。这两项保持为 49 与当前顶层文件数，不手改回 30 和 26。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | cowork/ledger/round-001/issues/p2-9/03-discussion/v1-sign-pm.md |
| RD | 同意 | 无 | v1 | cowork/ledger/round-001/issues/p2-9/03-discussion/v1-sign-rd.md |
| 技术负责人 | 同意 | 无 | v1 | cowork/ledger/round-001/issues/p2-9/03-discussion/v1-sign-tech-lead.md |
