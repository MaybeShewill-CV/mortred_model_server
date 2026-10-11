# 第3步签字

- 问题: P1-9
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: 相对已选定的 v2，只把 centerface 的 shapes 从 `input:1x3x640x640` 改为 `input.1:1x3x640x640`。尺寸仍是 `1x3x640x640`。QA 与实现说明里，本机这份 engine 的输入名是 `input.1`；旧名字 `input` 会让该次 `trtexec` 找不到张量、没有 `Throughput:`，表 5 因此不发表。验收 5 要求该行命令含契约表里的 `--shapes=`，测试义务写明 centerface 的命令含 `--shapes=input.1:1x3x640x640`。读表 5 和 `docs/benchmark-h3/centerface.txt` 就能判定。其余命令、验收 2 的硬件三段，以及 39 个都成功才发表表 5，与 v2 相同。验收 1 到 9 仍编号，黑盒可判定。不改 HTTP 契约，不把 engine 当成可拷到别的机器，也不写入训练或业务策略。
