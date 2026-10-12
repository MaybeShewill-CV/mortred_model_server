# 实现说明

- 方案: `cowork/ledger/round-001/issues/p0-1/03-design-final.md`
- 方案版本: v2
- 分支: `cowork/r1-p0-1-libface`
- worktree: `G:\Codex\mortred_wt_r1_p0_1`（WSL：`/mnt/g/Codex/mortred_wt_r1_p0_1`）
- commit: `6faa446aea51d32e5e5a8a9989f3237e6558797c`
- 实现偏差: 缺陷来自方案。验收标准 7 要求整份 `python3 scripts/product_static_bs1.py --check` 退出码为 0，失败来自方案没有列入改动面的两份 CPU toml 和脚本里原有的 MNN 检查。只改 libface 尺寸守卫那一句，不能让这两项通过。

## 行为变化

这一轮没有改代码，也没有提交。`6faa446` 里的 libface 尺寸守卫已经要求 `model_input_image_size = [480, 640]`。`--check` 在这句之后仍会读 `mobilenetv2_cpu.toml` 和 `resnet_cpu.toml`，并要求同时含 `type = "mnn"` 与 `device = "cpu"`。这两份文件现在是 `type = "onnx"`、`device = "cpu"`，所以 `validate_product_files` 仍追加两条错误，`main` 返回 1。热路径没有动：推理尺寸已经由 toml 的 `[480, 640]` 决定，这次失败发生在检查脚本里，不在 JPEG 推理路径上。

## 文件

无。

## 验证

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `cd /mnt/g/Codex/mortred_wt_r1_p0_1 && python3 scripts/product_static_bs1.py --check` | 1 | stderr 只有 `mobilenetv2_cpu.toml` 与 `resnet_cpu.toml` 必须保持 MNN on cpu。没有 libface 尺寸错误。 |

## 未做

没有改 MNN 检查，没有改这两份 CPU toml，没有扩大 diff，没有提交。基线仓库、`weights`、`3rd_party`、`_lib` 都没有动。

## 新增问题

无。
