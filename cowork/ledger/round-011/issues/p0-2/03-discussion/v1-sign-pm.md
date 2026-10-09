# 第3步签字

- 问题: P0-2
- 方案版本: v1
- 结论: 不同意
- 条件: 不设置 `MORTRED_PACK`。冻结名单中每一个 id，在该 id 的产品 onnx 已位于 `conf/product_static_bs1.json` 的 `onnx` 路径且本机有 trtexec 时，执行该 id 这次失败输出里的 `next:` 命令 `mortredctl prepare --id <该 id>`，退出码为 0，表内该 `engine` 存在且大小大于 0；执行前其余 id 的产品 engine 若不存在，执行后仍不存在。`yolov8n`、`yolov8l`、`yolov8x` 各自执行，一次 `yolov8s` 生成不代替这三条。缺该 id 的 onnx、缺 trtexec 或缺 GPU 时，该 id 记挂起，写明缺的环境和未跑的验收，未生成不得记成通过。
- 理由: 第 1 条到第 5 条、第 9 条到第 11 条能黑盒判定退出码、engine 路径和 `next:`。冻结条目还要求每一个 id 执行该行后，本机生成该 id 的产品 engine。第 6 条到第 8 条只覆盖 `yolov8s`、`metric3d_512`、`yolov8n`，其余 36 个 id 没有这项编号验收。
