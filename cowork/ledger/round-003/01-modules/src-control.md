# 模块评分

- id: src-control
- 权重: 15
- 分数: 9.3
- 读过的路径: `src/control/supervisor.h`、`src/control/supervisor.cpp`、`src/control/catalog.h`、`src/control/gateway/gateway_app.h`、`src/control/gateway/gateway_app.cpp`

## 已具备

- 一 catalog id 一 OS 进程；supervisor 管启停与状态；gateway 鉴权、转发、`/metrics`。
- `fork` 之后、`exec` 之前只用 async-signal-safe 路径（已关闭 ST5 的修复仍在）。
- 未鉴权与错误 token 不消耗随后合法请求配额（ST2）；单模型停驻后该 id 503，其他 id 仍成功（ST4）。
- token 缺一或 metrics 与推理 token 相同则拒绝启动（OP4）；停止信号有序退出（OP5）。
- PF4：五段 gateway 耗时与 `server_time_ms` 写入 `/metrics`。第 2 轮关闭的 P2-19 修复仍在：`render_stage_metrics` 在 deque 滚动样本上计算 p50，**不再**用 `stage_observed_ <= 4096` 关掉 `mortred_gateway_stage_dominant`（`gateway_app.cpp` `record_stage_sample` / `render_stage_metrics`）。

## 问题

无

## 范围外

- 已关闭 P2-19 的旧陈述不重新入列；当前源码在超过 4096 次成功后仍写出 dominant。
- 已关闭的 ST2/ST4/ST5/OP4/OP5 重复描述不重新入列。

## 依据

分数 9.3：上一轮因 P2-19 停在 9.0；该缺口已合入关闭，本模块无未关闭问题。完成标准落在本模块的 ST2、ST4、ST5、ST6、OP2–OP5、PF4 对得上当前源码。不到 10.0，因未在本机重跑超过 4096 次的 metrics 验收（未验证判断，不写成故障）。

## 未覆盖

无
