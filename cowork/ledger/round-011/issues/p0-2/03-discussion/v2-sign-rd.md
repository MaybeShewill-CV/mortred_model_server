# 第3步签字

- 问题: P0-2
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `scripts/mortredctl_doctor.sh` 在 pack 占用率与 pack TensorRT 段之后、活体检查之前已有 `GATE_FAIL` 退出。未设置 `MORTRED_PACK` 时，默认 pack `conf/packs/demo.toml` 只有 CPU MNN，无 TensorRT 的 pack 段可以保持成功。产品 engine 门只读 `conf/product_static_bs1.json`：表内 39 个 id 与冻结名单一致，engine 路径互不为子串；`yolov8s`、`metric3d_512` 的 fp16 路径，以及 fp32 六条 `mobilenetv2`、`resnet`、`densenet`、`bisenetv2`、`attentive_gan`、`realesrgan`，与方案点名一致。文件不存在或大小为 0 时打印该相对路径和 `next: mortredctl prepare --id <id>`，退出码 1，并停在活体检查之前。已设置 `MORTRED_PACK` 时不进入这道门，现有 `next: mortredctl prepare --pack` 保持不变。`src/control/cli/cli_app.cpp` 已把 `prepare` 的剩余参数原样交给 `scripts/mortredctl_prepare.sh`，不必改 `src/`。`--id` 可在入口与显式 `--pack` 或名单外 id 一起直接退出 1。名单内 id 必须绕开 demo pack 无 TensorRT 时的提前退出，走现有 `--write-manifest` 与 `convert_trt_engines.sh --only`：完整 engine 相对路径含 `/`，按 engine 字段精确匹配，因此 `yolov7` 与 `yolov7x`、`yolov8s` 与 `yolov8n`/`yolov8l`/`yolov8x` 不会互相转换；随后只检查该文件存在且非空，并跳过 pack 模型列表和 `/ready`。大小为 0 的文件会被现有转换视为已存在而跳过；方案已规定转换前去掉该文件，或传入现有 `--force`。onnx 缺失时转换脚本只跳过并退出 0，随后的非空检查使该命令退出码为 1。验收 1 到 12 都已编号，可由退出码、合并输出里的路径与整行 `next: mortredctl prepare --id <id>`、以及文件大小一次判定；`--id` 后的 id 按整词核对，`yolov7x` 不是 `yolov7`。第 6 到 8 条和第 12 条在缺该 id 的 onnx、缺 trtexec 或缺 GPU 时按方案挂起，未生成不得记成通过。
