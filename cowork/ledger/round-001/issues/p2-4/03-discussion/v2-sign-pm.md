# 第3步确认

- 问题: P2-4
- 方案版本: v2
- 结论: 不同意
- 条件: 标准 3 对 `yolov8s`、`libface` 和键 `gateway` 这三处的 `started_at_ms` 只确认各有一个 JSON 整数。该整数大于两条 `start` 时，标准 3 仍然通过。验收不把该整数与进程真实启动时刻、`before_sample`、两条 `start` 或两条 `end` 比较。标准 7 里「其中一个 pid 已经消失」只指这三处。`servers` 里其余 catalog id 的 `pid` 小于等于 0，或 `ready` 不为 true，不使标准 7 失败。同一份 json 不另设一段叙述。第 5 条的四次比较与这三处 `pid` 相等已经成立时，标准 7 通过。正文里的比较是这四次，不是五次。`after_sample` 不参与这四次。
- 理由: 验收标准 1、2、4、5、6 已编号，一次打开标准输出和 `logs/resident_pair_yolov8s_libface.json` 就能判定。上一版要求的六行标准输出、键名、`[pack.` 次数和成功响应也都落在这几条里。标准 3 和标准 7 不能一次判定。标准 3 写这三处的 `started_at_ms` 仍是进程启动时间，标准 5 又写它大于两条 `start` 时只有第 5 条仍然可以成立。这份状态正文里的 `started_at_ms` 是进程启动的 unix 毫秒，`before_sample` 与两条 `start`、`end` 是另一只单调时钟的整数。机器刚启动时，unix 毫秒会大于这两条 `start`。一次验收可以把它判成标准 3 失败，也可以只把它当成不参与比较的整数。标准 7 末句要确认文件里没有「其中一个 pid 已经消失」的另一段记录。同一次 `GET /api/v1/status` 的 `servers` 会带上其余 catalog id，未拉起的 `pid` 不是正数。这句话既可以只约束 `yolov8s`、`libface` 和键 `gateway`，也可以把其余 id 的 pid 算进去；后一种会把正确记录判失败。标准 5 写四次比较，「RD 不得自行决定」写五次比较，第五次没有名字。一次验收确认不了要不要多比一次。
