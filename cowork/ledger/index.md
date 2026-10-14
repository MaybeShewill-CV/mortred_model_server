# Ledger

- 基线分支: sub-agent-co-work
- 轮次上限: 20
- 当前轮次: 1
- 当前步骤: 4
- 当前问题: P2-11
- 阻塞: 无
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10
- 本轮起点加权分: 7.1
- 记录目录: cowork/ledger/round-001/
- 运行环境: 本机 WSL Ubuntu（/mnt/g/Codex/mortred_model_server）
- 状态: stopped
- 停止原因: P2-11 第 4 条未通过。技术负责人同意停住：官方 `--raw` 字节是 `demo_data/model_test_input/feature_point/test.png`，进不了 `jpeggpu`，而 `decode_auto` 已写成 `gpu`。在不得改 demo、解码路径、16 张投票网格、手填或放宽落点的限制里，第 4 条到不了退出码 0。这不是第 4 条通过，没有合入。清单仍为 6 行。`realesrgan` 与 `metric3d_512` 没有跑。

## 轮次

| 轮次 | 复核加权分 | 未关闭 P0 | 未关闭 P1 | 未关闭 P2 | 结果 |
| --- | --- | --- | --- | --- | --- |
