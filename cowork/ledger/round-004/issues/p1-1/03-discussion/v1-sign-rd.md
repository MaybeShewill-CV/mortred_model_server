# 第3步签字

- 问题: P1-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: drain 可注入时限可在现有头文件落地；worker_pool 单测注入 200ms 可在 2 秒内返回；`drain timeout` 日志可用 LogSink 断言。
