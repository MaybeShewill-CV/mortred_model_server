# 第3步签字

- 问题: P0-2
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的审查条件已写入第 12 条、测试义务和 39 行自测。不设置 `MORTRED_PACK` 时，产品 engine 门在 pack 段之后、活体检查之前核对冻结名单 39 个 id 的存在与非空；缺失则退出码 1，点名 `conf/product_static_bs1.json` 的 `engine`，并给出 `next: mortredctl prepare --id <id>`。`mortredctl prepare` 的剩余参数交给 `scripts/mortredctl_prepare.sh`。`--id` 与 `--pack` 同时出现，或 id 不在这 39 个之中，入口退出码 1。名单内 id 用完整 engine 相对路径走现有 `--only` 精确匹配，只生成该 id，再检查存在且大小大于 0。第 1 条到第 12 条都能用退出码、合并输出和文件大小一次判定。`yolov8n`、`yolov8l`、`yolov8x` 各自有生成命令。缺该 id 的 onnx、缺 trtexec 或缺 GPU 时该 id 挂起，未生成不算通过。已设置 `MORTRED_PACK` 时仍只检查该 pack。demo pack、产品表、产品 toml、`src/`、HTTP 与 tensor 保持不动，范围也不扩到 P0-3、P1-7 至 P1-12、D-8 至 D-12。
