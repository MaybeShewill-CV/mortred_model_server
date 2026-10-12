# 模块评分

- id: ci
- 权重: 5
- 分数: 9.0
- 读过的路径: `cowork/ledger/rubric.md`、`cowork/ledger/index.md`、`.github/workflows/ci.yml`、`.github/workflows/codeql.yml`、`.github/workflows/release.yml`、`.github/actionlint.yaml`、`.github/dependabot.yml`、`.github/pull_request_template.md`、`scripts/ci_eval_inference_paths.py`、`scripts/ci_print_tier_summary.py`、`scripts/check_contract_sync.py`、`scripts/gen_openapi.py` 的 `--check` 路径（1149–1184 行）

## 已具备

- OpenAPI 与仓库的一致性由两条始终排队的作业检查。`quality-gates` 执行 `python3 scripts/gen_openapi.py --check`（`.github/workflows/ci.yml` 264–272）。`--check` 在生成文本与 `docs/openapi.json`、`src/server/openapi_doc.h` 不一致，或 catalog 示例不合法时返回 1（`scripts/gen_openapi.py` 1162–1182）。`cpu-profile` 没有 `if`、没有 `needs`，构建 `contract_dump` 后执行 `scripts/check_contract_sync.py`（`ci.yml` 451–456）。该脚本比对 C++ catalog 的新 dump 与 `docs/contract_dump.json`，再调用 `gen_openapi.py --check`（`scripts/check_contract_sync.py` 44–71）。
- GPU 作业被跳过时，push 和 pull_request 的结论是失败，并写明没有 TensorRT 证明。`gpu-pr-gate`、`gpu-nightly-full` 只在 `vars.MORTRED_HAS_GPU_RUNNER == 'true'` 且事件匹配时排队（`ci.yml` 829–833、902–906）。`inference-paths` 在 push / pull_request 上以 `if: always()` 运行，把 `needs.cpu-profile.result` 和 `needs.gpu-pr-gate.result` 传给 `scripts/ci_eval_inference_paths.py`（`ci.yml` 1022–1036）。cpu 为 `success`、gpu 为 `skipped` 时，脚本打印 `NOT_A_TRT_PROOF` 和 “This result is not a TensorRT or CUDA proof.”，返回 2（`scripts/ci_eval_inference_paths.py` 54–63）。`TRT_PROOF=yes` 只在 gpu 为 `success` 时打印（同文件 64–66）。该 job 没有 `continue-on-error`。`deploy-tools` 执行这两个脚本的 `--self-test`（`ci.yml` 577–578）；self-test 固定 `("success", "skipped", 2)`，并要求 stdout 含 `NOT_A_TRT_PROOF`、`SKIPPED`、`NOT RUN` 和免责声明，且不含 `TRT_PROOF=yes`（`scripts/ci_eval_inference_paths.py` 142、165–175）。
- schedule 与 `workflow_dispatch` 上，`t2-tier-notice` 把 `needs.gpu-nightly-full.result == skipped` 印成 `T2 ... SKIPPED`，并写 “Not a full-zoo claim.”（`ci.yml` 1000–1020，`scripts/ci_print_tier_summary.py` 80–82）。`cpu-profile` 的作业说明写明该 job 不证明 TensorRT 或 CUDA（`ci.yml` 358–362）。
- 同一工作流还有与 UX3 无关、但已接上的门闩：gcc 的 tests-only `check` 与至少 5 个 valgrind 用例（`ci.yml` 54–125）；`-Werror` 与变更行 clang-format（178–257）；TSAN 标签测试和 ASan/UBSan 的 tests-only `check`（312–345）；`MORTRED_BUILD_PROFILE=cpu` 的全量构建、sha256 锁定的 hosted golden，以及 `verify_deployment.sh --basic`（420–459）；部署脚本 `bash -n` / `py_compile`、mock trtexec 的 TensorRT 10 标志契约、actionlint（554–677）。`.github/actionlint.yaml` 登记 self-hosted 标签 `gpu`。

## 问题

无

## 范围外

- Docker 镜像与默认运行用户。`container-boot`、`full-werror`、`full-build-pr`、`nightly`、`codeql.yml` 的 full 模式，以及 `release.yml` 的 ghcr 镜像和 tarball，都在这条里。
- 多机调度、Kubernetes、Windows、macOS。
- 覆盖集以外的 catalog id。`MORTRED_GPU_SMOKE_FILTER` 不因此扩成全目录。

## 依据

落在本模块的完成标准行是 UX3。当前文件里两条可观察标准都有门闩，本模块没有未满足行。

- UX3 前半已满足：CI 检查 OpenAPI 与仓库一致。证据是「已具备」第一条。`quality-gates` 与 `cpu-profile` 都没有用 `if` 跳过该步骤。
- UX3 后半已满足：GPU 作业跳过时写出的是 `NOT_A_TRT_PROOF` 或 `SKIPPED` / “Not a full-zoo claim.”。push / pull_request 的退出码是 2。`TRT_PROOF=yes` 只在 `gpu-pr-gate` 结果为 `success` 的分支出现。

PF1–PF5、AV1、ST1–ST6、TP1–TP3、UX1、UX2、UX4、OP1–OP5、EX1–EX3 的可观察对象是压测报告、进程行为、README、产品 toml 或运维命令，不在 `.github/` 的工作流结论里。这些行留在各自模块，不在本记录里改记为未满足或范围外。

分数 9.0 的出处是 UX3 的两道门闩都在 `ci.yml` 中，且 `deploy-tools` 用 `--self-test` 锁住跳过路径的退出码和文案。未到 10.0 的出处是：`http_infer_rps.py` 在该工作流里只出现 `--self-test`（`ci.yml` 272）；`gpu-pr-gate` 成功时执行的是 `MORTRED_GPU_SMOKE_FILTER` 的四个 golden（`ci.yml` 47–48、874–883）。本次调用没有跑压测。性能、吞吐、engine、占用行不因这次未跑 GPU 而标成阻塞。

## 未覆盖

无
