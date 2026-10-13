# 第3步确认

- 问题: P2-9
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1–4 已编号，且都能在外部判定。`--check` 只在名单顺序和顶层文件字节哈希都一致时打印 `zero drift:` 并返回 0；`case_names` 含 `dinov2_vits14` 与 `metric3d_512`，`golden_sha256` 含对应两个 json 键，键值等于当时文件字节的 SHA-256。整表 `--record` 是这条锁定句要退出码 0 的落地，只改 `test/golden_baseline.json`。不改检查脚本、golden 字节、gateway、supervisor 和 TensorRT context，也不扩到 P2-10、P2-11。两条自测都在 `/mnt/g/Codex/mortred_model_server`，期望退出码都是 0。上一版审查意见无。
