# 关闭记录

- 轮次: 2
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11、P2-12
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
| docs | 5 | 7.4 | P2-12。英文和中文模型服务教程里，覆盖集 `mobilenetv2`、`superpoint` 的 `--model` 与 catalog id、server toml 的 `model=` 不一致。按这两条命令启动会得到 `unknown --model`。 | P2-14。英文和中文模型服务教程要求在显存够时加大仓库自带配置的 `worker_nums`。`docs/deployment.md` 要求 `conf/server` 的 `worker_nums` 保持 1，校准结果写在机器 pack。P2-16。中文「添加新服务」示例的 id、`--model`、`model=` 与 catalog 不一致，且示例没有 family `"densenet"`。P2-17。英文和中文接口契约公共响应示例的 `model.name` 是 `MOBILENETV2`。`docs/openapi.json` 里 `mobilenetv2` 的响应示例是 `mobilenetv2`。 |
| ci | 5 | 8.0 | 无 | P2-20。`gpu-pr-gate` 成功时，`inference-paths` 打印 `TRT_PROOF=yes` 并以 0 退出。这次作业实际跑的四个 golden 都是 `type=onnx`。P2-21。`inference-paths` 在 `gpu-pr-gate` 为 success 时打印 `TRT_PROOF=yes`，不核对同一次 gtest 里有没有跑过的产品 `type=tensorrt` 用例。上一轮证据：`cowork/ledger/round-002/01-modules/ci.md`；`cowork/ledger/round-002/02-p0-p1-final.md` |

## 附录

本步新关闭的只有 P2-12。合入 `1566f90c2dcf1af532b9027ae7a06d7ee811aac8`。问题分支提交 `31f28c9d888702bda81215056e066a5423026640`。QA 结论 pass。重跑 `python3 /mnt/c/Users/MaybeSheWill-CV/AppData/Local/Temp/p2_12_qa.py`，退出码 0，打印 `p2-12-tutorial-model-ids ok`。diff 只有 `docs/tutorials_of_model_servers.md` 与 `docs/tutorials_of_model_servers.zh-cn.md`，各两处 `--model` 实参。当前两份教程里，分类命令的 `--model` 是 `mobilenetv2`，特征点命令的 `--model` 是 `superpoint`。

重复描述：英文和中文模型服务教程里，覆盖集 `mobilenetv2`、`superpoint` 的 `--model` 与 catalog id、server toml 的 `model=` 不一致。按这两条命令启动会得到 `unknown --model`。

P2-14、P2-16、P2-17 不因这次 diff 关闭。两份教程关于 `worker_nums` 的句子仍是原文。`docs/how_to_add_new_server.zh-cn.md` 仍是 `DENSENET`。两份接口契约的公共响应示例仍是 `"name": "MOBILENETV2"`。P2-19、P2-20、P2-21 的文件不在这次 diff 里。

P2-13、P2-15、P2-18 维持锁定附录里的并入，不单独开放。`--model YOLOV5`、`--model BISENETV2`、`--model ATTENTIVE_GAN_DERAIN` 仍在两份教程里。这三个 id 在覆盖集外，不入列，也不并进 P2-12。`mortred_gateway_stage_dominant` 的 `stage` 是否为五段里 p50 最大的名字，一次读取 `/metrics` 对不齐，不单列，也不写进 P2-19。

P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11 维持已关闭。陈述不重复，也不写入仍开放。P2-8 留在范围外。

九个模块分数沿用第 2 轮原文。没改到的模块沿用 `cowork/ledger/round-002/01-modules/`。

`cowork/ledger/round-002/discovered.md` 的 CI 残余有出处和现在的行为：`.github/workflows/ci.yml` 不引用两份模型服务教程，分类或特征点 `--model` 改回大写时 CI 仍会通过。它没有写出改完后一次外部验收即可确认的变化。「改回大写后 CI 仍通过」也不能靠对当前树的一次外部验收确认。不新入列。同一份附录里的真实启动，方案 v1 写明不启动 `mortred-model-server.out`。不新入列。

## 新入列

无

## 编排复核

P2-12 的问题陈述在「已关闭的问题陈述」和附录的重复描述里，不在「仍开放的问题」。六条未关闭陈述与 `02-p0-p1-final.md` 一致。新入列为无。不退回。工作树 `G:\Codex\mortred_wt_r2_p2_12` 已删除。临时脚本 `p2_12_qa.py` 已删除。
