# 技术方案
- 问题: P0-2
- 版本: v1
## 目标
在解析前强制 body 上限；空 Content-Length / chunked 不能绕过；`/jobs` POST 同样 413。
## 边界
- 不改 packing/Docker；不改默认 64MB 上限语义。
## 实现要点
1. `request_admission.h`：新增 `received_body_exceeds(req, limit_mb)`（CL 超限或 get_parsed_body size 超限）；空/非法 CL 仍用实际 body。
2. `serve_process`：在 content-type 分支解析之前调用，超限直接 413。
3. `async_endpoints`：configure 传入 limit；submit 路径同样检查。
4. 单测覆盖空 CL + 超限 body、合法 CL、/jobs 路径谓词。
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | Round 9 锁定 |
| cv-pm | 同意 | 无 | v1 | 用户锁定 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定 |
| cv-rd | 同意 | 无 | v1 | 用户锁定 |
