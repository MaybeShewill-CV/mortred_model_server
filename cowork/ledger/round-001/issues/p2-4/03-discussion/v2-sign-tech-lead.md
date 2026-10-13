# 第3步确认

- 问题: P2-4
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 上一版条件已经落进验收标准 1 至 7。`--check` 的退出码、六行标准输出，以及命令前后 `logs/resident_pair_yolov8s_libface.json` 都不存在，一次就能判定；实跑只打开这一份 json。`before` 与 `after` 各是一次 `GET /api/v1/status` 正文，采样整数在正文外。`yolov8s` 与 `libface` 只取 `servers` 里对应 `id`，gateway 只取键 `gateway` 且 `id` 为 `__gateway`。第三个 id 只数 `pack` 里的 `[pack.`。重叠只比较 `before_sample` 与两条 `start`、`end` 这五个整数，`started_at_ms` 仍留在状态正文里。成功只看 HTTP 200、根上 `status` 为 0、`results` 长度为 1、`results[0].status` 为 0。ops 新增 `scripts/resident_pair_record.py`，临时 pack 不写回 `conf/packs/`。test 按 `p0_1_engine_fail_closed` 注册 `resident_pair_record_rules`，只跑 `--check`，进入默认 `ctest -LE valgrind`；实跑不进默认 check，规则通过不能单独当成 TP2。现有 autostart、状态字段、端口、占用门闩和每个 worker 各自的 execution context 与正文一致。自测命令在本机 WSL `/mnt/g/Codex/mortred_model_server`，期望退出码已写明。
