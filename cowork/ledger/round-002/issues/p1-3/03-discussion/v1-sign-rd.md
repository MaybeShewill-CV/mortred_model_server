# 第3步签字

- 问题: P1-3
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 仅改 `scripts/mortredctl_prepare.sh` 即可：参数门禁、缺 pack 时输出 `step: pack-file` 与路径，通过后用非 `exec` 方式调用 `prepare_pack.sh`。四条自测均为 WSL bash，不需完整 GPU prepare。
