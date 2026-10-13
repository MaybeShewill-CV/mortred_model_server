# 第3步确认

- 问题: P2-9
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 4 已编号，在仓库根执行一次检查并打开账本就能判定。标准 1 在 `/mnt/g/Codex/mortred_model_server` 执行 `python3 scripts/golden_drift_check.py --check`，退出码为 0，标准输出包含 `zero drift:`。现有脚本在整份 `case_names` 顺序和 `test/golden/` 顶层每个文件的字节 SHA-256 都一致时打印这句并返回 0。标准 2 看 `test/golden_baseline.json` 的 `case_names` 含 `dinov2_vits14` 与 `metric3d_512`。标准 3 看同一文件的 `golden_sha256` 含 `dinov2_vits14.json` 与 `metric3d_512.json`。标准 4 把这两个键的值与这两个文件当时的原始字节 SHA-256 比较；退出码为 0 时，其余顶层文件的字节哈希同时与账本一致，任一文件字节与账本不一致则该命令退出码为 1。自测两条命令分别覆盖退出码，以及这两个名字和两个键。HTTP 路径、产品 toml、server toml 与 `mortredctl doctor` 保持现状。账本按现有 `--record` 记下已有文件的字节哈希，不重写 `test/golden/` 里的期望数值，不跑 GPU 推理，也不把产品配置改成 onnx。README 模型目录已有这两个 id，`docs/api-contract.md` 里 `metric3d_512` 的深度响应保持不动。
