# 模块评分

- id: ci
- 权重: 5
- 分数: 9.0
- 读过的路径: `.github/workflows/ci.yml`、`scripts/ci_eval_inference_paths.py`、`scripts/ci_container_boot.sh`、`scripts/convert_trt_engines.sh`（对照）、`conf/ci_hosted_golden.json`

## 已具备

- UX3：CI 检查 OpenAPI 与仓库一致；GPU 作业跳过时写出 `NOT_A_TRT_PROOF` / `SKIPPED`，push/PR 退出码 2。
- 第 2 轮关闭的 P2-20 / P2-21 修复仍在：`TRT_PROOF=yes` 仅当 `gpu == success` **且** `--product-trt yes` **且** `--gtest-xml` 证明至少有一个未 skip 的产品 `type=tensorrt` 用例。当前工作流 `inference-paths` 传 `--product-trt no`（`ci.yml`），因此成功路径打印 `TRT_PROOF=no`，不会把 ONNX-only 的 `MORTRED_GPU_SMOKE_FILTER` 读成 TensorRT 证明。`--self-test` 锁住跳过/无 XML/有 TRT XML 等组合。
- `ci_container_boot.sh` 在无默认 cpu 产品后改为每 boot 注入 cpu overlay，与 `ea22867d` 一致。

## 问题

无

## 范围外

- 已关闭 P2-20、P2-21 重复描述不重新入列。
- `MORTRED_GPU_SMOKE_FILTER` 仍是四个 ONNX hosted golden；在 `--product-trt no` 下这是明确的「非 TRT 证明」路径，不是未关闭缺口。
- 官方压测并发命令不在本工作流。

## 依据

分数 9.0：上一轮因成功分支可被误读成 TRT 证明而停在 8.0；门闩与 XML 证明已合入，本模块无未关闭问题。UX3 两条字面条件都有门闩。不到 10.0，因 GPU 冒烟过滤器本身仍是 ONNX 子集（需显式 `--product-trt yes` + 产品 TRT XML 才宣称证明）。

## 未覆盖

无
