# 第3步签字

- 问题: P1-4
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 析构在双 drain 返回后没有 destroying 日志，可插入；commit_workers 能构造已 init 状态。不改超时时长。
