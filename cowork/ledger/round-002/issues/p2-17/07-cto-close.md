# 关闭记录

- 轮次: 2
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11、P2-12、P2-14、P2-16、P2-17
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-002/01-modules/src-models.md` |
| src-server-apps | 20 | 9.4 | 无 | 无。上一轮证据：`cowork/ledger/round-002/01-modules/src-server-apps.md` |
| src-control | 15 | 9.0 | 无 | P2-19。同一次成功推理超过 4096 次，且 gateway 五段 p50 之和大于模型 `server_time_ms` p50 的一半时，gateway `/metrics` 不再写出 `mortred_gateway_stage_dominant`。上一轮证据：`cowork/ledger/round-002/01-modules/src-control.md`；`cowork/ledger/round-002/02-p0-p1-final.md` |
| src-factory-common | 10 | 9.2 | 无 | 无。上一轮证据：`cowork/ledger/round-002/01-modules/src-factory-common.md` |
| conf | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-002/01-modules/conf.md` |
| ops | 15 | 8.8 | 无 | 无。上一轮证据：`cowork/ledger/round-002/01-modules/ops.md` |
| test | 10 | 8.8 | 无 | 无。上一轮证据：`cowork/ledger/round-002/01-modules/test.md` |
| docs | 5 | 7.4 | P2-12。英文和中文模型服务教程里，覆盖集 `mobilenetv2`、`superpoint` 的 `--model` 与 catalog id、server toml 的 `model=` 不一致。按这两条命令启动会得到 `unknown --model`。P2-14。英文和中文模型服务教程要求在显存够时加大仓库自带配置的 `worker_nums`。`docs/deployment.md` 要求 `conf/server` 的 `worker_nums` 保持 1，校准结果写在机器 pack。P2-16。中文「添加新服务」示例的 id、`--model`、`model=` 与 catalog 不一致，且示例没有 family `"densenet"`。P2-17。英文和中文接口契约公共响应示例的 `model.name` 是 `MOBILENETV2`。`docs/openapi.json` 里 `mobilenetv2` 的响应示例是 `mobilenetv2`。 | 无 |
| ci | 5 | 8.0 | 无 | P2-20。`gpu-pr-gate` 成功时，`inference-paths` 打印 `TRT_PROOF=yes` 并以 0 退出。这次作业实际跑的四个 golden 都是 `type=onnx`。P2-21。`inference-paths` 在 `gpu-pr-gate` 为 success 时打印 `TRT_PROOF=yes`，不核对同一次 gtest 里有没有跑过的产品 `type=tensorrt` 用例。上一轮证据：`cowork/ledger/round-002/01-modules/ci.md`；`cowork/ledger/round-002/02-p0-p1-final.md` |

## 附录

本步新关闭的只有 P2-17。合入 `c22ff0e344614a4f68591e401d7b82360a96cd19`。问题分支提交 `9e2d125830f010bc1a973f89842b949289085719`。QA 结论 pass。重跑 `python3 /mnt/c/Users/MaybeSheWill-CV/AppData/Local/Temp/p2_17_rerun.py`（root 为已合入基线），退出码 0，打印 `p2-17-api-contract-name ok`。diff 只有 `docs/api-contract.md` 与 `docs/api-contract.zh-cn.md`，把公共响应示例 `model.name` 由 `MOBILENETV2` 改为 `mobilenetv2`。当前两份接口契约公共响应示例的 `model.name` 已与 openapi mobilenetv2 200 example 对齐。

重复描述：英文和中文接口契约公共响应示例的 `model.name` 是 `MOBILENETV2`。`docs/openapi.json` 里 `mobilenetv2` 的响应示例是 `mobilenetv2`。

P2-19、P2-20、P2-21 的文件不在这次 diff 里。陈述未变。

P2-13、P2-15、P2-18 维持锁定附录里的并入，不单独开放。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11、P2-12、P2-14、P2-16 维持已关闭。陈述不重复，也不写入仍开放。P2-8 留在范围外。

九个模块分数沿用第 2 轮原文。没改到的模块沿用 `cowork/ledger/round-002/01-modules/`。本步不重开、不重算加权分。

`cowork/ledger/round-002/discovered.md` 对 P2-17 写「无」。没有同时具备出处、现在的行为和一次外部验收即可确认的变化的新项。不新入列。

## 新入列

无

## 编排复核

P2-17 的问题陈述在「已关闭的问题陈述」和附录的重复描述里，不在「仍开放的问题」。仍开放为 P2-19、P2-20、P2-21，陈述与 `02-p0-p1-final.md` / 上一份关闭记录一致、未变。新入列为无。不退回。
