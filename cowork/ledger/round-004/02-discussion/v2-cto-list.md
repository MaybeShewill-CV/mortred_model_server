# 列表修订

- 轮次: 4
- 版本: v2
- 针对: v1

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 | 相对上轮 |
| --- | --- | --- | --- | --- | --- | --- |
| P1-1 | 成功 init 后停服对 worker/async drain 无界等待 | 小团队在 WSL 或本机停一个已经 init 的 catalog 进程，模型 worker 卡住 | 析构对 worker/async drain 无界等待，进程可能一直挂到外层 SIGKILL | 人为卡住 in-flight 后停服，在写明的时限内结束，或日志给出可观测的超时终态 | `src/server/base_server_impl.h` 析构；`src/server/async_endpoints.h` drain | 新证据 |
| P1-2 | 深度族缺不依赖权重的假 tensor 后处理契约测试 | WSL 上改深度后处理或 shape，手头没有权重 | 默认 ctest 没有「假 tensor → 输出或失败点名」的深度门，缺权重时相关门会跳过 | 默认 ctest 在无权重时仍跑深度假 tensor 用例；shape 或后处理错误时失败信息能定位到后处理或 shape | `src/models/mono_depth_estimation/`；`test/CMakeLists.txt` | 新证据 |
| P1-3 | 已有 fail-closed 黑盒脚本未进默认 ctest | 小团队合入前只跑默认 ctest | `scripts/test_p0_1_engine_fail_closed.sh` 可手跑，但默认 ctest 不包含它 | 默认 ctest 出现该脚本对应的用例；无 GPU 时 SKIP 的退出码约定不变 | `scripts/test_p0_1_engine_fail_closed.sh`；`test/CMakeLists.txt` | 新证据 |

## 附录
未升：WORKFLOW 首次线程池锁定、分段耗时仍依赖 touch 文件、golden 缺权重 SKIP。既有：`trt_engines.json` 旧名、OCR 代码无识别、install-host 不 start、FAQ 游离 `</details>`、无 GPU CI。
