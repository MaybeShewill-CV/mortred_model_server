# Round 8 discovered

外部只读源码审计（基线 `10d3a764`）的发现项，按处置分三类。

## 进入本轮队列（P0）

- P0-1: `_bin/server_e2e_contract_test` 45 例 2 例 FAIL（`rate_limit_qps=1` 的 429 断言 vs 模型侧限流已移除）；`python3 scripts/check_consistency.py` 退出 1（`openapi_doc.h` 已是 runtime loader，`gen_openapi.py` 仍写 embedding 版本）；`test/CMakeLists.txt` 7 个 sanitizer 标签 vs `ci.yml` 构建 6 个目标。
- P0-2: `/ready` = 有空闲 worker，满载即 503 并把健康实例摘出入口。
- P0-3: 请求体大小检查在解析之后；`/jobs` 无检查；无 `Content-Length` 时跳过。
- P0-4: `receive_timeout` 未设置，滴流连接可占满 200 连接入口。
- P0-5: 日志每行 flush + reaper 线程 join + `wait_shutdown` 无超时 → 子进程不回收、stop 无效、SIGTERM 后不退出。
- P0-6: `libface` 按 /32 padding 预处理却按未 padding 尺寸反解，非方形图的框与关键点系统性偏移。

## 进入本轮队列（P1）

- P1-1: `clip_boxes` 无效；`min_box_area_px` 坐标空间随检测器变化，测试名与代码相反。
- P1-2: 管理/抓取凭据被每个模型子进程继承（`/proc/<pid>/environ` 可读）。
- P1-3: 只 kill 单 pid，无进程组，孙进程存活（OP5）。
- P1-4: 数值门禁可跳过：部分通过返回 0，缺权重一律 skip，报告不写执行数。
- P1-5: `tensor.h` 的 `CHECK` 在请求路径上 abort 整个模型进程。
- P1-6: `model_run_timeout<=0` 只警告，导致无界等待与永不回复。
- P1-7: `/metrics` 的 method 标签取自未鉴权的原始方法字符串，基数无界。

## 进入本轮队列（P2）

- P2-1 至 P2-9: golden 裸整数 JSON 533MB；`rate_limit_qps` 在 44 份配置中无效；死代码（`do_work` 等）；异步结果深拷贝与无界保留；stuck worker 无指标；`model_config_file_path` 相对 CWD；无响应压缩；`contract_baseline` 无消费者；`/metrics` 被推理 token 挡住。

## 未入列（处置见 `02-p0-p1-final.md` 附录）

- rubric 明文范围外: CLIP 分词器非 BPE；SAM AMG 并行解码数据竞争；SAM encoder 输入 BGR；LDM latent scale 取反；`pp_matting` alpha dtype/插值不一致；容器 root 用户；单进程一模型结构与 370MB／61 线程基线；Windows/macOS；MNN/ORT 吞吐；动态 shape/batch；多机与 K8s；`msocrnet`。
- 已由第 7 轮关闭（不重复入列）: `max_queue`/`worker_budget` 解析、queue residency、hard `worker_budget`、NVML probe、FixedWindow 产品成员、`verify_gpu_gate_evidence` 入 CI、CI version 独立步。
- 需用户决定是否扩范围: 依赖 sha256 与离线安装；产物 `LICENSE`/`NOTICE` 与被捆绑 TensorRT/cuDNN 的再分发许可、SBOM 覆盖二进制；内置 TLS；审计日志。
- 一次外部验收确认不了: 结构化日志与 trace 贯通；其余参数语义偏差清单（需逐个模型定夹具）；golden 容差是否过松（需参考实现对比）。
