# 第3步签字

- 问题: P1-9
- 方案版本: v1
- 结论: 不同意
- 条件: 验收 2 的硬件单元格按英文逗号分成恰好三段，依次是 GPU 名称、显存、驱动版本；三段去掉首尾空白后都非空，显存含 `MiB` 或 `GiB`。整格不含 `name, memory.total`、`memory.total`、`driver_version`、`compute_cap`、`N/A`。现有「机器」节里的 `name, memory.total [MiB], memory.free [MiB], driver_version, compute_cap` 不能通过。
- 理由: 验收 1、3 到 9 能对上 39 行、命令、`trtexec_rps`、日期和记录文件里的 `Throughput:`。验收 2 的硬件只排除空单元格、整格等于 `name, memory.total`、以及含 `N/A`。现有机器节的表头 `name, memory.total [MiB], memory.free [MiB], driver_version, compute_cap` 整格并不等于 `name, memory.total`，也不含 `N/A`，按这条会通过；写成「见机器节」也会通过。只看这一格，判定不了行上写出的是 GPU 名称、显存和驱动版本。
