# 技术方案

- 问题: D-11
- 版本: v1
- 对应列表: `cowork/ledger/round-012/02-p0-p1-final.md` 的 D-11

## 边界

只做已经冻结的 D-11。冻结句不改写：`deployment.md` 把 cpu 目录写成 `resnet50`，并把 `msocrnet` 写成 Bench-only。

改完后能观察到的变化只这一句：这两处 cpu 名称是 `resnet`，Bench-only 列表不再包含 `msocrnet`。

本条没有上一版方案，没有待并入的审查条件。

现在 `docs/deployment.md` 第 108 行的 cpu 格是 `curated: mobilenetv2, resnet50`，同一行 Bench-only 列表是 `CLIP, LightGlue, FastSAM, SAM prompt, MsOcrNet`。第 253 行是 `models (cpu: the two `*_cpu` entries - mobilenetv2 / resnet50).`。分类 catalog id 是 `resnet`：cpu 产品 toml 为 `type = "mnn"`、`device = "cpu"`，cpu 服务配置 `model="resnet"`，服务路径是 `/mortred_ai_server_v1/classification/resnet`。权重文件名仍是 `resnet-50.mnn`。场景分割 catalog id 是 `msocrnet`，登记在 HTTP `catalog()` 里，`bench_catalog()` 为空；产品 toml 为 `type = "tensorrt"`、`device = "gpu"`。cpu 服务配置只有 `mobilenetv2` 与 `resnet` 两份。

不覆盖 P0-2、P1-8、D-8、D-12。不重开已关闭原句：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、P1-12、D-1 至 D-6、D-9、D-10。P1-5 只含 `yolov8s`。P1-11 只含 `bisenetv2`、`pphuman_mobile`、`modnet`、`ppmatting_512` 的 CPU 全分辨率掩码重合。`centerface` 的 engine 输入名 `input.1` 不重开。

## 做法

### 推荐

只改 `docs/deployment.md` 里已经点名的两行，各做一次原地替换，不增行、不删行。

第 108 行只动两处字符：cpu 格的 `resnet50` 改为 `resnet`；Bench-only 列表删掉 `, MsOcrNet`，句号留在 `SAM prompt` 后面。该行其余字符保持原样，包括 HTTP catalog 括号里的 `seg`，以及 `CLIP`、`LightGlue`、`FastSAM`、`SAM prompt`。

第 253 行只把 `resnet50` 改为 `resnet`。`the two`、`*_cpu`、`mobilenetv2` 保持原样。

改完后第 108 行全文是：

`| **Models** | HTTP catalog (classification / detection / OCR (`dbnet` text detection only, no recognition) / seg / matting / enhancement / SuperPoint / depth / DINOv2 / SAM AMG / diffusion). Bench-only: CLIP, LightGlue, FastSAM, SAM prompt. No MOT. RT-DETR is unimplemented. | curated: mobilenetv2, resnet |`

改完后第 253 行全文是：

`models (cpu: the two `*_cpu` entries - mobilenetv2 / resnet).`

代价：只动这一个文件的两行。不构建，不拉起进程，不占显存，不生成 engine。`docs/deployment.zh-cn.md` 与 `README.md` 里仍写着 `resnet50` 的那两处句子留在本条以外。

### 否决

连同 `docs/deployment.zh-cn.md`、`README.md` 的精选句，或把第 108 行 HTTP catalog 括号改写成全体 id 名单。冻结句只点名 `deployment.md` 的这两处。扩出去之后，一次黑盒看到的就不再是队列里的那两行。这条否决。

把 catalog id 改成 `resnet50`，或把 `msocrnet` 从 HTTP `catalog()` 挪进 bench catalog，让现有句子变成对的。id 已经是 `resnet` 与 `msocrnet`。改注册会动 HTTP 路径和产品 toml，并碰到挂起的 P1-8。这条否决。

## 改动面

- `docs/deployment.md` 第 108 行：cpu 格改为 `curated: mobilenetv2, resnet`；Bench-only 列表改为 `CLIP, LightGlue, FastSAM, SAM prompt`。
- `docs/deployment.md` 第 253 行：cpu 名称改为 `resnet`，该行全文如上。

## 明确不改

- `docs/deployment.zh-cn.md`、`README.md`、`README.zh-cn.md`、`docs/benchmark-set-a.md`、`docs/benchmark-set-a.zh-cn.md`、`docs/unsupported-boundaries.md`。
- `docs/deployment.md` 除第 108 行、第 253 行以外的全部行，以及这两行上未被点名的字符。
- 分类与场景分割的源码、`catalog()`、`bench_catalog()`、产品 toml、cpu 服务配置、权重路径 `weights/classification/resnet/resnet-50.mnn`。
- `msocrnet` 的 toml、engine、黄金、数值用例。本条不把它写入 cpu 精选名单。
- HTTP 与张量契约，gateway `:8080`，supervisor `:8787`，一个 catalog id 一个 OS 进程。
- 产品 toml 的 `type=mnn` / `type=tensorrt`。gpu profile 仍是 CUDA 12.x + TensorRT 10.x。engine 仍在本机本卡用 `mortredctl prepare` 生成。cpu profile 仍不链 TensorRT。
- P0-2、P1-8、D-8、D-12，以及已关闭原句。

## 契约

没有 HTTP 契约变化，没有张量契约变化，没有 toml 变化。catalog id 仍是 `resnet` 与 `msocrnet`。`resnet` 的 cpu 产品行仍是 `type=mnn`、`device=cpu`。`msocrnet` 的产品行仍是 `type=tensorrt`、`device=gpu`。

文档上的变化只有两处：第 108 行与第 253 行的 cpu 名称是 `resnet`；第 108 行 Bench-only 列表是 `CLIP, LightGlue, FastSAM, SAM prompt`，其中没有 `msocrnet`。

## 性能与资源

不新增常驻进程，不占用 gateway 或 supervisor，不占显存，不生成 engine。cpu 精选仍是 `mobilenetv2` 与 `resnet` 两个 id。`msocrnet` 仍不在 cpu 精选里。

## 验收标准

两条都在 Linux 上、仓库根目录执行。通过条件是脚本退出码 0、标准输出与下面的原文一致、标准错误为空。不读源码，不读 toml，不启动进程。

验收 1。第 108 行与第 253 行全文。退出码 0。命令：

```bash
python3 - <<'PY'
from pathlib import Path
import sys
lines = Path("docs/deployment.md").read_text(encoding="utf-8").splitlines()
if len(lines) < 253:
    sys.exit(1)
row = lines[107]
note = lines[252]
expect_row = "| **Models** | HTTP catalog (classification / detection / OCR (`dbnet` text detection only, no recognition) / seg / matting / enhancement / SuperPoint / depth / DINOv2 / SAM AMG / diffusion). Bench-only: CLIP, LightGlue, FastSAM, SAM prompt. No MOT. RT-DETR is unimplemented. | curated: mobilenetv2, resnet |"
expect_note = "models (cpu: the two `*_cpu` entries - mobilenetv2 / resnet)."
sys.stdout.write(row + "\n" + note + "\n")
if row != expect_row or note != expect_note:
    sys.exit(1)
PY
```

标准输出原文：

```
| **Models** | HTTP catalog (classification / detection / OCR (`dbnet` text detection only, no recognition) / seg / matting / enhancement / SuperPoint / depth / DINOv2 / SAM AMG / diffusion). Bench-only: CLIP, LightGlue, FastSAM, SAM prompt. No MOT. RT-DETR is unimplemented. | curated: mobilenetv2, resnet |
models (cpu: the two `*_cpu` entries - mobilenetv2 / resnet).
```

验收 2。第 108 行 Bench-only 列表。退出码 0。命令：

```bash
python3 - <<'PY'
from pathlib import Path
import sys
lines = Path("docs/deployment.md").read_text(encoding="utf-8").splitlines()
if len(lines) < 108 or "Bench-only:" not in lines[107]:
    sys.exit(1)
bench = lines[107].split("Bench-only:", 1)[1].split(".", 1)[0].strip()
sys.stdout.write(bench + "\n")
if bench != "CLIP, LightGlue, FastSAM, SAM prompt" or "msocrnet" in bench.casefold():
    sys.exit(1)
PY
```

标准输出原文：

```
CLIP, LightGlue, FastSAM, SAM prompt
```

## 测试义务

QA 跑验收 1、验收 2。通过与否只看这两条命令的退出码、标准输出原文和空的标准错误。不把读 `catalog()`、读 toml、打开 README、启动 gateway `:8080`、启动 supervisor `:8787`、拉起模型、生成 engine、跑 `msocrnet` 数值比较写成通过条件。

## 自测命令

与验收标准里的两条命令逐字相同。在 Linux 上、仓库根目录分开执行。期望退出码都是 0。标准错误都为空。标准输出原文与对应验收相同。

验收 1：

```bash
python3 - <<'PY'
from pathlib import Path
import sys
lines = Path("docs/deployment.md").read_text(encoding="utf-8").splitlines()
if len(lines) < 253:
    sys.exit(1)
row = lines[107]
note = lines[252]
expect_row = "| **Models** | HTTP catalog (classification / detection / OCR (`dbnet` text detection only, no recognition) / seg / matting / enhancement / SuperPoint / depth / DINOv2 / SAM AMG / diffusion). Bench-only: CLIP, LightGlue, FastSAM, SAM prompt. No MOT. RT-DETR is unimplemented. | curated: mobilenetv2, resnet |"
expect_note = "models (cpu: the two `*_cpu` entries - mobilenetv2 / resnet)."
sys.stdout.write(row + "\n" + note + "\n")
if row != expect_row or note != expect_note:
    sys.exit(1)
PY
```

标准输出原文：

```
| **Models** | HTTP catalog (classification / detection / OCR (`dbnet` text detection only, no recognition) / seg / matting / enhancement / SuperPoint / depth / DINOv2 / SAM AMG / diffusion). Bench-only: CLIP, LightGlue, FastSAM, SAM prompt. No MOT. RT-DETR is unimplemented. | curated: mobilenetv2, resnet |
models (cpu: the two `*_cpu` entries - mobilenetv2 / resnet).
```

验收 2：

```bash
python3 - <<'PY'
from pathlib import Path
import sys
lines = Path("docs/deployment.md").read_text(encoding="utf-8").splitlines()
if len(lines) < 108 or "Bench-only:" not in lines[107]:
    sys.exit(1)
bench = lines[107].split("Bench-only:", 1)[1].split(".", 1)[0].strip()
sys.stdout.write(bench + "\n")
if bench != "CLIP, LightGlue, FastSAM, SAM prompt" or "msocrnet" in bench.casefold():
    sys.exit(1)
PY
```

标准输出原文：

```
CLIP, LightGlue, FastSAM, SAM prompt
```

## RD 不得自行决定

- cpu 名称的拼写。两处都是 `resnet`。不得写成 `resnet50`、`ResNet`、`resnet-50`、`resnet_50` 或 `RESNET`。
- 不得按权重文件名 `resnet-50.mnn` 把文档改回 `resnet50`，也不得改该权重路径。
- Bench-only 列表的成员与顺序。删掉的只有 `, MsOcrNet`。`CLIP`、`LightGlue`、`FastSAM`、`SAM prompt` 留在列表里。不得改成 `msocrnet`、`MsOcrNet`、`MSOCRNet` 或其他大小写后再留在该列表。
- 不得把 `msocrnet` 写进第 108 行的 cpu 精选，也不得在 `docs/deployment.md` 里新增含 `msocrnet` 或 `MsOcrNet` 的句子。HTTP 侧沿用该行已有的 `seg`。
- 不得改第 253 行的 `two` 与 `*_cpu`。cpu 精选仍是 `mobilenetv2` 与 `resnet`。
- 不得增删 `docs/deployment.md` 的行，使第 108 行或第 253 行换到别的行号。
- 不得改 `docs/deployment.zh-cn.md`、`README.md`、`README.zh-cn.md`、基准文档、`docs/unsupported-boundaries.md`、源码、toml、测试、脚本、镜像或 engine。
- 不得把本条做成 P0-2、P1-8、D-8、D-12，或已关闭原句的重开。
