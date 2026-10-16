# 技术方案
- 问题: P0-4
- 版本: v1
## 目标
supervisor 不因日志 I/O 停顿永久挂死：LogBuffer 不再每行 flush；handle_exit 先关 pipe 再 join 且不堵死 reaper；wait_shutdown 有界。
## 边界
- 不改 packing/Docker；不改重启策略语义。
## 实现要点
1. `LogBuffer::append`：去掉每行 `_file.flush()`；仅 rotate/dtor（及可选批量阈值）flush。
2. `handle_exit`：先 close out/err fd，再 join reader；join 放到独立 joiner（reaper 不阻塞），完成后清 readers_live。
3. `wait_shutdown`：`wait_for` 30s 上限，超时仍 `_threads_stop` 并 join。
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | Round 9 锁定 |
| cv-pm | 同意 | 无 | v1 | 用户锁定 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定 |
| cv-rd | 同意 | 无 | v1 | 用户锁定 |
