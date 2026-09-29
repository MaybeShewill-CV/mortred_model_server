# 第2步表决

- 轮次: 4
- 针对版本: v1
- 结论: 待修订

## 逐条
无

## 不同意的条目
空 P0/P1 无法再抬综合分。请从附录升最多 3 条为新 P1，不要升 OCR 识别、install-host 默认 start、空 scrape、GPU CI、已关闭 P0-1 主路径，也不要只做 FAQ 或 catalog nullptr：

1. 同进程 `WORKFLOW_library_init` 只认首次线程池。后起实例要么按自身配置生效，要么启动即失败并点名已锁定的首次配置。
2. 成功 init 后析构对 worker/async drain 无界等待。停服须在可写明的时限内结束，或给出可观测超时终态。
3. 推理 golden 缺权重时默认 SKIP 后整门仍绿。缺权重应 fail-closed 并点名，或交付主族有不依赖实权重、在 WSL 必跑的断言。
