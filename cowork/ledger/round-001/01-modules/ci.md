# 模块评分

- id: ci
- 权重: 5
- 分数: 7.2
- 读过的路径: `.github/workflows/ci.yml`

## 优势
- PR/push 有路径过滤触发，并串起 tests（`check` + Valgrind fail-closed）、quality-gates（`-Werror`、增量 clang-format、一致性/OpenAPI）、sanitizers（TSAN 标签门 + ASan/UBSan 全量 tests-only），合入前能拦住大量代码层回归（`.github/workflows/ci.yml` jobs `tests` / `quality-gates` / `sanitizers`）。
- `cpu-profile` 做 cpu 全量编译、sha256 锁定的 hosted golden（分类/检测/OCR/关键点/分割）、契约同步与 `verify_deployment.sh --basic`，`inference-paths` 要求 `cpu-profile` 必须 success，无 GPU runner 时仍有可定位的推理活路门禁（同文件 `cpu-profile`、`inference-paths`）。
- `deploy-tools` 校验部署脚本语法、manifest、mock `trtexec` 契约与 actionlint；`full-build-pr` 用 Docker 全量 `-Werror` 并在失败时把编译错误打成 `::error::` 注解，便于小团队定位（同文件 `deploy-tools`、`full-build-pr`）。

## 劣势
- GPU 黄金烟测依赖仓库变量 `MORTRED_HAS_GPU_RUNNER=true` 与自托管 `[self-hosted, X64, gpu]`；未配置时 `gpu-pr-gate` 被 skip，`inference-paths` 仍可通过——典型中小团队合入前拦不住真实 TRT/CUDA 交付回归（同文件 `gpu-pr-gate`、`inference-paths`、环境注释 L45–48 / L820–825）。
- PR 默认只跑 gcc 行；clang、clang-tidy、Docker full-werror 多依赖 `workflow_dispatch`/`full-matrix` 或 schedule，且 clang-tidy 为 `continue-on-error` 信息性任务，合入前门禁覆盖不完整（同文件 `tests` matrix、`clang-tidy`、`full-werror`、`nightly`）。
- `container-boot` 仅在 Docker 相关路径变更或 push main 时跑 compose 活路；非容器 diff 的 PR 可能漏掉 gateway/compose 启动与 infer 回归（同文件 `container-boot` paths-filter 与条件步骤）。

## 依据
- 分数 7.2：以 `.github/workflows/ci.yml` 为准——CPU/单测/契约/部署脚本门禁扎实且失败可上传 log / 注解，但交付主路径（GPU + TRT engine）在默认小团队配置下合入前非 fail-closed，达不到「交付相关回归必在合入前红且可定位」的 9+ 水准。

## 未读
无
