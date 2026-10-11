# 实现说明

- 方案: P1-11 技术方案（`03-design-final.md`）
- 方案版本: v2，采用 2026-10-10 选定的技术负责人条件
- 分支: `cowork/r11-p1-11-mask-overlap`
- 工作树: `G:\Codex\mortred-cowork-r11-p1-11`
- commit: `2dceffc1b3e45a3c99d4f0e379ab1a6146433c19`
- 偏离: 无

## 行为变化

`compare_scene_segmentation` 与 `compare_matting` 不再调用 `expect_fingerprint`。分割比较全分辨率 `segmentation_result`，类型必须是 `CV_32SC1`。对黄金里出现过的每个类别算 IoU，再取平均。抠图比较 `matting_result`，接受 `CV_8UC1` 或 `CV_32SC1`，前景是值 `>= 128` 的像素，比较前景 IoU。平均或前景重合 `>= 0.90` 通过。只有重合 `< 0.90` 时，失败输出才含 `mask overlap` 和 `0.90`。尺寸不同、分割类别少于两个、抠图前景少于 2 的失败输出不含这两个字。类型不符的失败路径同样不含这两个字。

四份黄金由对应 CPU onnx 在现有输入图上生成，写成 `test/golden/<用例名>.json`。`bisenetv2_segmentation` 有 11 个类别。`pphuman_segmentation` 有 2 个类别。`modnet_matting` 与 `ppmatting_matting` 的前景像素数和值小于 128 的像素数都大于 2。四份 `.png` 已删除。`test/golden_baseline.json` 只把这四个键从 png 换成 json，并写入新文件原始字节的 sha256。`case_count` 仍是 30，`golden_count` 仍是 26。

服务热路径没有改。比较仍在现有测试进程的一次 `run` 之后，不增加推理拷贝、色彩往返或第二套进程。

这次回修没有改代码。QA 点名的 `test/golden/mobilenetv2_classification.json` 不在 `2dceffc1` 的改动里。它的 git blob 仍是 `16f80dabd4fb69c5ad9031be5fa56917fdbef70a`，与 HEAD、与 `806813aa` 相同。当前工作区该文件没有 CR，1004 个 LF，sha256 就是基线的 `72a51e13c2fe533d12681dec6bb03ea99bd038928648a28ec538e90d3e9ab9fd`。四份新 json 的工作区哈希也与基线一致。

## 文件

- `test/model_golden_registry.h`
- `test/golden/bisenetv2_segmentation.json`
- `test/golden/pphuman_segmentation.json`
- `test/golden/modnet_matting.json`
- `test/golden/ppmatting_matting.json`
- `test/golden_baseline.json`
- 删除 `test/golden/bisenetv2_segmentation.png`、`test/golden/pphuman_segmentation.png`、`test/golden/modnet_matting.png`、`test/golden/ppmatting_matting.png`

## 验证

回修在工作树根目录重跑了漂移检查。验收 1 到 7、验收 9 仍是 QA 的 pass，这次没有重跑。前两行是已有提交的自测。

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `MORTRED_UPDATE_GOLDEN=1 _bin/model_golden_test --gtest_filter=model_golden.bisenetv2_segmentation:model_golden.pphuman_segmentation:model_golden.modnet_matting:model_golden.ppmatting_matting` | 0 | 四条都打印 `golden updated`，标准输出不含 `NUMERICAL_PASSED` |
| `_bin/model_golden_test --gtest_filter=model_golden.bisenetv2_segmentation:model_golden.pphuman_segmentation:model_golden.modnet_matting:model_golden.ppmatting_matting` | 0 | `[  PASSED  ] 4 tests.`，标准输出含 `NUMERICAL_PASSED` |
| `python scripts/golden_drift_check.py --check` | 0 | `zero drift: 30 cases and 26 golden files unchanged` |

工作树干净，HEAD 仍是 `2dceffc1b3e45a3c99d4f0e379ab1a6146433c19`。没有新提交。

## 未做

未改 `test/model_golden_test.cc` 的测试名、配置和输入图。未改 `expect_fingerprint` 的其他调用。未改产品 toml、HTTP、模型输出类型、engine 生成。未构建 `full`。未跑 `msocrnet`，未跑 19 条同进程过滤。未执行 `golden_drift_check.py --record`。验收 6、7、9 的改黄金负例留给测试。

未改 `mobilenetv2_classification.json` 的 git 对象，未把基线哈希改成 CRLF 字节，未改 `scripts/golden_drift_check.py`，未加 `.gitattributes`。

## 新发现

无
