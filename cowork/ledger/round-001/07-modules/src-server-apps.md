# 模块评分

- id: src-server-apps
- 权重: 15
- 分数: 7.7
- 读过的路径: 上轮记录；本轮 diff 未触及 `src/server` / `src/apps`

## 优势
- 单进程按 catalog id 启动，配置错误在启动前退出。
- HTTP 推理面含鉴权、健康检查、同步 POST 与统一拒绝。
- 异步任务有 jobs 生命周期。

## 劣势
- `WORKFLOW_library_init` 同进程只认第一次线程池配置。
- 成功 init 后析构对 worker drain 无界等待。
- 同步路径超时与部分成功难在现场拆段。

## 依据
- 分数 7.7：沿用上轮。本轮 diff 未改该模块。

## 未读
本轮未重读，沿用上轮记录。
