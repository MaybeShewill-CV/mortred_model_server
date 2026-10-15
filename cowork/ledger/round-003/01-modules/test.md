# 模块评分

- id: test
- 权重: 10
- 分数: 8.8
- 读过的路径: `test/` 目录、`test/golden/`、`test/contract_baseline/`、`test/golden_baseline.json`、`test/model_golden_test.cc`、`test/backend_unittest.cc`、相关 server e2e / contract 用例、`docs/perf/coverage_set_http_rps.md`、`conf/ci_hosted_golden.json`

## 已具备

- 契约、鉴权、队列打满、batch=1 观测有真实 server 用例（PF3、ST1–ST3）。
- 覆盖集九个 id 有进程内 golden 或 TensorRT product 用例；`conf/ci_hosted_golden.json` 中 mobilenetv2 指向 `conf/ci/mobilenetv2_onnx_hosted.toml`（不再依赖已删除的产品域 `mobilenetv2_cpu.toml`）。
- `docs/perf/coverage_set_http_rps.md` 九行与 `docs/benchmark-h3/<id>.txt` Throughput 对得上（TP1）。
- `ea22867d` 将 resnet cpu golden 归到 `conf/ci/resnet_cpu_golden.toml`，`model_golden_test.cc` 引用已更新。

## 问题

无

## 范围外

- 已关闭 P2-4、P2-9 重复描述不重新入列。
- 默认 `check` 不跑官方压测；缺权重时 golden 可跳过数值比对。

## 依据

分数 8.8：失败语义与已写 HTTP 吞吐有测试/报告锁住。本模块无未关闭问题。未执行 `golden_drift_check.py --check`，不把退出码写成已跑过的 0。

## 未覆盖

无
