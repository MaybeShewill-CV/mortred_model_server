# 第3步签字

- 问题: P0-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `scripts/mortredctl_doctor.sh` 在 pack 占用率与 pack TensorRT 段之后、活体检查之前已有 `GATE_FAIL` 退出。未设置 `MORTRED_PACK` 时，默认 pack `conf/packs/demo.toml` 只有 CPU MNN，无 TensorRT backend 的 pack 段可以保持成功。产品 engine 门只读 `conf/product_static_bs1.json`：冻结名单与表内 39 个 id 一致，engine 路径互不为子串；`yolov8s`、`metric3d_512` 的 fp16 路径，以及 fp32 六条 `mobilenetv2`、`resnet`、`densenet`、`bisenetv2`、`attentive_gan`、`realesrgan`，与方案点名一致。文件不存在或大小为 0 时打印该相对路径和 `next: mortredctl prepare --id <id>`，退出码 1，并停在活体检查之前。已设置 `MORTRED_PACK` 时不进入这道门，现有 `next: mortredctl prepare --pack` 保持不变。`mortredctl prepare` 的剩余参数已经交给 `scripts/mortredctl_prepare.sh`。`--id` 可在入口与 `--pack` 或名单外 id 一起直接退出 1。名单内 id 走现有 `--write-manifest` 与 `convert_trt_engines.sh --only`：完整 engine 相对路径含 `/`，按 engine 字段精确匹配，只转换该 id，再检查存在且非空，并跳过 pack 模型列表和 `/ready`。大小为 0 的文件会被现有转换视为已存在而跳过；要满足写出后再检查非空，实现时先去掉该空文件，或传入现有 `--force`。onnx 缺失时转换脚本只跳过，不下载权重；随后的非空检查使该命令退出码为 1。验收 1 到 11 都可由退出码、合并输出中的路径与 `next:` 行、以及文件大小一次判定。第 6 到 8 条在缺 onnx、trtexec 或 GPU 时按方案挂起。
