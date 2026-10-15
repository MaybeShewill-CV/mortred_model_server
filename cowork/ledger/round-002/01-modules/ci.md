# 模块评分

- id: ci
- 权重: 5
- 分数: 8.0
- 读过的路径: `cowork/ledger/rubric.md`、`.github/workflows/ci.yml`、`scripts/ci_eval_inference_paths.py`、`scripts/ci_print_tier_summary.py`、`scripts/check_contract_sync.py`、`scripts/gen_openapi.py` 的 `--check` 路径（1149–1183 行）、`scripts/check_consistency.py` 的 OpenAPI 检查与 `main`（234–281、1133–1214 行）、`scripts/check_hosted_golden.py` 的 `check_ci_inference_contract`、`scripts/convert_trt_engines.sh` 的已有引擎分支（371–374 行）、`conf/ci_hosted_golden.json` 的 `gpu_smoke`、`conf/ci/yolov8_onnx_hosted.toml`、`conf/ci/libface_onnx_hosted.toml`、`conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml`、`test/model_golden_test.cc` 中上述四个用例和产品 TensorRT 用例、`test/model_golden_registry.h` 的 `load_model_cfg`、`docs/ci-golden-regression.md`

## 已具备

- OpenAPI 与仓库的一致性由两条会排队的作业检查。`quality-gates` 执行 `python3 scripts/check_consistency.py` 和 `python3 scripts/gen_openapi.py --check`（`.github/workflows/ci.yml` 264–272）。`check_consistency.py` 核对每个 `server_uri` 是否出现在 `docs/openapi.json`，并核对 `src/server/openapi_doc.h` 是否嵌入这份 JSON（234–281、1139–1140 行）。`gen_openapi.py --check` 在生成结果与 `docs/openapi.json`、`src/server/openapi_doc.h` 不一致，或 catalog 示例不合法时返回 1（1162–1182 行）。`cpu-profile` 没有 `if`、没有 `needs`，构建 `contract_dump` 后执行 `scripts/check_contract_sync.py`（`ci.yml` 451–456）。该脚本比对 C++ catalog 的新 dump 与 `docs/contract_dump.json`，再调用 `gen_openapi.py --check`（`check_contract_sync.py` 44–71 行）。
- GPU 作业被跳过时，push 和 pull_request 的结论是失败，并写明没有 TensorRT 证明。`gpu-pr-gate`、`gpu-nightly-full` 只在 `vars.MORTRED_HAS_GPU_RUNNER == 'true'` 且事件匹配时排队（`ci.yml` 829–833、902–906）。`inference-paths` 在 push / pull_request 上以 `if: always()` 运行，把 `needs.cpu-profile.result` 和 `needs.gpu-pr-gate.result` 传给 `scripts/ci_eval_inference_paths.py`（1022–1036 行）。cpu 为 `success`、gpu 为 `skipped` 时，脚本打印 `NOT_A_TRT_PROOF` 和 “This result is not a TensorRT or CUDA proof.”，返回 2（`ci_eval_inference_paths.py` 54–63 行）。该 job 没有 `continue-on-error`。`deploy-tools` 执行 `--self-test`（`ci.yml` 576–578）；self-test 固定 `("success", "skipped", 2)`，并要求 stdout 含 `NOT_A_TRT_PROOF`、`SKIPPED`、`NOT RUN` 和免责声明，且不含 `TRT_PROOF=yes`（142、165–175 行）。
- schedule 与 `workflow_dispatch` 上，`t2-tier-notice` 把 `needs.gpu-nightly-full.result == skipped` 印成 `T2 ... SKIPPED`，并写 “Not a full-zoo claim.”（`ci.yml` 1000–1020，`ci_print_tier_summary.py` 80–82）。`cpu-profile` 的作业说明写明该 job 不证明 TensorRT 或 CUDA（`ci.yml` 358–362）。
- 同一工作流还有与 UX3 字面两条无关、但已接上的门闩：gcc 的 tests-only `check` 与至少 5 个 valgrind 用例（54–125 行）；`-Werror` 与变更行 clang-format（178–257 行）；TSAN 标签测试和 ASan/UBSan 的 tests-only `check`（312–345 行）；`MORTRED_BUILD_PROFILE=cpu` 的全量构建、sha256 锁定的 hosted golden，以及 `verify_deployment.sh --basic`（420–459 行）；部署脚本 `bash -n` / `py_compile`、mock trtexec 的 TensorRT 10 标志契约、actionlint（554–677 行）。`check_hosted_golden.py` 要求 `MORTRED_GPU_SMOKE_FILTER` 与 `conf/ci_hosted_golden.json` 的 `gpu_smoke.cases` 顺序和名字一致（232–241 行）；当前两边都是下面这四个名字。

## 问题

1. `gpu-pr-gate` 成功时，`inference-paths` 打印 `TRT_PROOF=yes` 并以 0 退出。这次作业的 `--gtest_filter` 是 `model_golden.yolov8_detection`、`model_golden.yolov8_mixed_size_batch_matches_single_runs`、`model_golden.mobilenetv2_classification`、`model_golden.libface_detection`（`ci.yml` 45–48、874–883）。这四个用例加载的配置是 `conf/ci/yolov8_onnx_hosted.toml`、`conf/model/classification/mobilenetv2/mobilenetv2_cpu.toml`、`conf/ci/libface_onnx_hosted.toml`，`backend.type` 都是 `onnx`，权重是 `.onnx`（`test/model_golden_test.cc` 58、189、197–198、236 行）。`load_model_cfg` 会改写已有的 `device`（`test/model_golden_registry.h` 151–160 行）。产品 TensorRT 用例 `Yolov8TrtDetection.MatchesCpuGoldenBoxes` 和 `TrtProductGolden.*` 不在该过滤器里（`model_golden_test.cc` 466 行起）。`convert_trt_engines.sh --only yolov8` 在引擎文件已存在且未加 `--force` 时打印 skip 并继续（371–374 行）。工作流注释写 “Eight cases”（`ci.yml` 45、825 行），可执行过滤器是这四个名字。改完后一次外部验收即可确认的变化：只跑这四个 ONNX 用例、yolov8 引擎步骤被 skip 或只写出引擎文件时，`inference-paths` 日志里没有 `TRT_PROOF=yes`。日志出现 `TRT_PROOF=yes` 时，同一次 `gpu-pr-gate` 的 gtest XML 里至少有一个产品 `type=tensorrt` 用例，且 `skipped=0`。

## 范围外

- Docker 镜像与默认运行用户。`ci.yml` 里的 `container-boot`、`full-werror`、`full-build-pr`、`nightly` 在这条里。
- 多机调度、Kubernetes、Windows、macOS。
- 覆盖集以外的 catalog id。`MORTRED_GPU_SMOKE_FILTER` 不因此扩成全目录。
- 重复描述：P2-11。本工作流对 `scripts/server/http_infer_rps.py` 只执行 `--self-test`（`ci.yml` 270–272），没有按已部署 `worker_nums` 的两倍并发跑 `--raw` 压测。这次调用没有跑压测，不把 PF1、PF5、TP1 标成阻塞，也不把这句再写入问题。

## 依据

落在本模块的完成标准行是 UX3。两条字面条件在当前文件里都有门闩。成功路径另有一条未关闭问题，所以问题不是「无」。

- UX3 前半已满足：CI 检查 OpenAPI 与仓库一致。证据是「已具备」第一条。`quality-gates` 与 `cpu-profile` 都没有用 `if` 跳过该步骤。
- UX3 后半已满足：GPU 作业被跳过时写出的是 `NOT_A_TRT_PROOF`，以及 `SKIPPED` / “Not a full-zoo claim.”。push / pull_request 的退出码是 2。`TRT_PROOF=yes` 只出现在 `gpu-pr-gate` 结果为 `success` 的分支（`ci_eval_inference_paths.py` 64–66 行）。
- 问题 1 是这条成功分支的现行行为。`gpu == "success"` 即打印 `TRT_PROOF=yes`，而该作业实际调度的四个 golden 是 `type=onnx`。产品 `type=tensorrt` 用例留在 `#ifdef MORTRED_HAS_TRT` 下，不进入 `MORTRED_GPU_SMOKE_FILTER`。

PF1–PF5、AV1、ST1–ST6、TP1–TP3、UX1、UX2、UX4、OP1–OP5、EX1–EX3 的可观察对象是压测报告、进程行为、README、产品 toml 或运维命令，不在 `.github/workflows/ci.yml` 的跳过结论里。这些行留在各自模块。本次调用没有跑压测，不把性能、吞吐、engine、占用行标成阻塞。官方压测并发是已部署 `worker_nums` 的两倍；本工作流没有这条命令。

分数 8.0 的出处：UX3 的跳过路径和 OpenAPI 门闩都在，并且 `deploy-tools` 用 `--self-test` 锁住跳过时的退出码和文案。问题 1 使一次 GPU 作业成功可以被读成 TensorRT 证明，而同一次 gtest 跑的是四个 ONNX 用例。这个模块可以挡住「跳过的 GPU 作业被写成 TensorRT 证明」，还不能把 `gpu-pr-gate` 的成功当成这台卡上的 TensorRT 证明。

## 未覆盖

无
