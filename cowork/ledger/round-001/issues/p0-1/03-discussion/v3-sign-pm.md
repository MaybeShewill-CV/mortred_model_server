# 第3步确认

- 问题: P0-1
- 方案版本: v3
- 结论: 不同意
- 条件: 在受测仓库根执行这一条命令，退出码为 0：打开 `scripts/product_static_bs1.py`，libface 尺寸守卫在 toml 不含 `model_input_image_size = [480, 640]` 时记失败，并运行 `python3 scripts/product_static_bs1.py --check`。这次 `--check` 的 stderr 不含 `libface params are not 640x480`。stderr 出现 `mobilenetv2_cpu.toml: cpu profile config must stay MNN on cpu` 与 `resnet_cpu.toml: cpu profile config must stay MNN on cpu`，且这次 `--check` 退出码不为 0 时，标准 7 仍然通过。
- 理由: 验收标准 1 至 6 仍能一次判定。验收标准 7 不能。它要求「执行一条命令」且退出码为 0，正文没有点名这条命令，同一条又写不采用 `python3 scripts/product_static_bs1.py --check` 的退出码。按该命令名执行时退出码为 1，stderr 是这两份 CPU 配置的 `cpu profile config must stay MNN on cpu`；两份文件现在是 `type = "onnx"`、`device = "cpu"`。末句「必须保持 MNN on cpu 的输出不使本条失败」既可以读成这两份文件必须仍是 MNN on cpu，也可以读成这段 stderr 不算失败。一次验收确认不了按哪一种通过。
