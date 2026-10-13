# 第3步确认

- 问题: P2-9
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `--record` 只把 `collect()` 写进 `test/golden_baseline.json`。`collect()` 只读 `test/model_golden_test.cc` 和 `test/golden/` 顶层文件的字节，baseline 也不在该目录内，所以 golden 文件本身不会被改。只读核对与方案一致：正则得到 49 个名字，顺序含 `dinov2_vits14` 与 `metric3d_512`，不含 `TRT_PRODUCT_GOLDEN` 的测试名；顶层是 56 个 json 加 6 个 png，共 62 个文件，含这两个 json 和 18 个 `trt_*.json`，没有 `msocrnet.json`。相对现账本只有 19 个 `cases added`、36 个 `golden file added`，以及 9 个 `golden file modified`（`densenet121_classification.json`、`dinov2_feature_embedding.json`、`enlightengan_enhancement.png`、`libface_detection.json`、`mobilenetv2_classification.json`、`resnet50_classification.json`、`yolov5_detection.json`、`yolov6_detection.json`、`yolov7_detection.json`）。没有 `cases removed`、`golden file removed` 或 `case declaration order changed`。按方案仍执行 `--record`，这 9 个键改为当前字节哈希，文件保持原字节。写入后的账本与同一次 `collect()` 相同，随后 `--check` 打印 `zero drift:` 并返回 0。第二条自测能读到这两个名字和两个 json 键，退出码也是 0。WSL 里是 Python 3.10.12，两条命令可在 `/mnt/g/Codex/mortred_model_server` 按原文执行。
