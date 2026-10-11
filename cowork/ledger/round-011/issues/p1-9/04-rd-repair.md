# 实现说明

- 方案: P1-9 `03-design-final.md`
- 方案版本: v2
- 分支: cowork/r11-p1-9-h3-bench
- 工作树: G:\Codex\mortred-cowork-r11-p1-9
- commit: 76309292c020951abfbf992081dcdefe153fb78a（本次未提交）
- 偏离: 表 5 和 `docs/benchmark-h3` 仍然没有发表。签字命令对 centerface 使用 `--shapes=input:1x3x640x640`，engine 路径是 `weights/object_detection/centerface_detection/centerface.engine.static_bs1.fp16`。该 engine 的优化剖面是 `Model input input.1 (profile 0): min=1x3x64x64, opt=1x3x640x640, max=1x3x768x768`。尺寸 `1x3x640x640` 与 opt 一致，被拒绝的是名字 `input`：带这条 `--shapes` 时标准错误是 `Cannot find input tensor with name "input"`，随后 `Inference set up failed`。已有采集里该进程退出码 1，标准输出没有 `Throughput:`。脚本按 v2 在退出码非 0 且没有 `Throughput:` 时不写残缺表。要让表 5 出现，只能把这条 `--shapes` 的张量名从 `input` 改成 `input.1`。shapes、engine 路径和 `--duration=15` 保持方案原文。影响是验收 1 到 7 和检查命令仍然不成立，39 行基准没有写入文档。

## 行为变化

代码保持 `76309292`。`h3_lines.py` 对 centerface 仍执行方案表里的 shapes 单元格 `input:1x3x640x640`，`.fp16` 仍加 `--fp16`，时长仍是 `--duration=15`。39 个 id 都在本次进程输出里打出 `Throughput:` 才写 `## 表 5 — 一行基准` 和 `docs/benchmark-h3/<id>.txt`。GPU、`3rd_party/bin/trtexec` 和 centerface engine 都在，该 id 的进程失败，表 5 与记录目录仍不存在。

推理热路径没有改。这一行基准只读已经放在 `3rd_party/bin/trtexec` 的二进制和已有 engine，现有推理路径不参与指标。

## 文件

无。`git status --short` 为空，HEAD 仍是 `76309292c020951abfbf992081dcdefe153fb78a`。

## 验证

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `python scripts/bench_set_a/check_h3_lines.py` | 1 | 工作目录 `G:\Codex\mortred-cowork-r11-p1-9`。标准错误是 `table 5 heading count 0`。表 5 不在文档里。这条检查没有启动 `trtexec`。 |
| `python scripts/bench_set_a/h3_lines.py` | 未跑 | 没有重跑 39 条 `trtexec`。已有采集里 `run 15/39 centerface` 的下一行是 `FAIL centerface rc=1`，这段没有 `Throughput:`，结束前标准错误含 `not published` 和 `centerface`。 |

## 未做

表 5 和 39 个 `docs/benchmark-h3/<id>.txt` 仍未写入。缺陷来自契约表 centerface 行的 shapes 单元格 `input:1x3x640x640`，以及做法里这一句：「下表「shapes」非空时，再加 `--shapes=` 接该单元格。除此之外不加别的参数。」engine 输入名是 `input.1`，这条命令在本机被拒绝。边界接着要求：「命令非 0、没有 `Throughput:`、engine 文件不在、或硬件探测失败时，不写该 id 的指标，也不写表 5。39 个都成功才发表表 5。」「RD 不得自行决定」写明不得改 engine 路径和四条 shapes。本次因此停在偏离点，没有改这些参数，也没有提交。

## 新发现

无。
