# 技术方案

- 问题: D-3 缺权重时 golden 默认仍 SKIP
- 版本: v2
- 对应列表: `cowork/ledger/round-010/02-p0-p1-final.md` 劣势 D-3

## 边界

上一版审查意见：PM 对 v1 不同意。条件是验收第 8 条改为两份文档分开判定，自测对每个文件单独检索；只命中 `NUMERICAL_NOT_RUN` 和 `77`、没有原文「单条缺权重仍可以跳过」或 `case skips` 时，第 8 条不通过。`test/ci_require_weights.h` 不含 `stays green` 单独判定。验收第 9 条改为只看 `gpu-nightly-full` 里步骤名含 `smoke-8` 的那一步：该步在 `unset MORTRED_CI_REQUIRE_WEIGHTS` 之前仍含 `MORTRED_CI_REQUIRE_WEIGHTS=1`；从这句 `unset` 到该步第一次 `gpu-rest-skips.json` 之前，同时含 `77`、`gpu-rest.xml` 和 `--allow-skips`。同文件更早位置的 `MORTRED_CI_REQUIRE_WEIGHTS=1` 不能算这一步通过。本版把该条件写入验收第 8 条、第 9 条和对应自测命令。不扩大到其他问题。

只覆盖已经冻结的这一条：本机没有权重时跑 golden，结果要能分开「未跑数值」和「已通过」。场景是一次 `model_golden_test` 里，选中的用例全部因为缺权重没有做数值比对。

现在的路径是：`test/model_golden_registry.h` 的 `run_case`，以及 `test/model_golden_test.cc` 里各手写用例，在 `weights_available` 为假时调用 `MORTRED_SKIP_OR_FAIL_WEIGHTS("weights not available")`。`test/ci_require_weights.h` 在未设置 `MORTRED_CI_REQUIRE_WEIGHTS`（或值为空、`0`、`false`、`FALSE`）时走 `GTEST_SKIP`。gtest 把全部跳过当成进程成功，退出码为 0。CTest 因此把名为 `model_golden_test` 的那一项印成 Passed。`MORTRED_CI_REQUIRE_WEIGHTS=1` 时同一位置改为 `GTEST_FAIL`。`MORTRED_UPDATE_GOLDEN=1` 在写出基线之后的 `GTEST_SKIP` 文案是 `golden updated`，与缺权重无关。

无权重时，直接跑该二进制的退出码改为 77，标准输出有 `NUMERICAL_NOT_RUN`。CTest 该项显示为 Skipped。已通过仍是退出码 0，且标准输出有 `NUMERICAL_PASSED`。`check` 在没有别的失败项时，总退出码仍可以是 0；区分点是这一项的结果文字，以及二进制自己的退出码和那一行标记。

不涉及 P0-1、P1-1 至 P1-6、D-1、D-2、D-4 至 D-6。

## 做法

### 推荐

只改 `model_golden_test` 这个进程在用例全部跑完、并且 gtest 已经写完本次结果（含 `--gtest_output` 的 xml）之后的结束状态。单条缺权重仍走现有宏：未设置 `MORTRED_CI_REQUIRE_WEIGHTS` 时，该条仍是 `GTEST_SKIP`，原文仍是 `weights not available`。xml 里这些条仍记为 skipped。

结束状态按本次选中的用例计：

- 选中条数大于 0，失败数为 0，每一条都是缺权重跳过，没有任何一条做完数值比对：进程退出码 77，标准输出恰好一行 `NUMERICAL_NOT_RUN`。不打印 `NUMERICAL_PASSED`。
- 失败数为 0，且每一条选中用例都做完数值比对：进程退出码 0，标准输出恰好一行 `NUMERICAL_PASSED`。不打印 `NUMERICAL_NOT_RUN`。
- 失败数大于 0：进程退出码 1。两行标记都不打印。`MORTRED_CI_REQUIRE_WEIGHTS=1` 且缺权重属于这一类。
- 同一进程里既有做完比对的用例，又有缺权重跳过，且失败数为 0：进程退出码 0。两行标记都不打印。
- 选中用例全部是 `golden updated` 跳过：进程退出码 0。两行标记都不打印。输出里仍有 `golden updated`。
- 过滤没有选中任何用例：不使用退出码 77，两行标记都不打印。退出码保持 gtest 对「没有选中用例」的原有结果。

CTest 里只给名为 `model_golden_test` 的测试设置 `SKIP_RETURN_CODE` 77。该项因此显示为 Skipped。退出码 1 仍是 Failed。退出码 0 仍是 Passed。

`gpu-nightly-full` 里，`unset MORTRED_CI_REQUIRE_WEIGHTS` 之后那一次剩余 golden（负向 filter，写出 `gpu-rest.xml`）若得到 77，该步继续把这份 xml 交给现有的 `scripts/ci_assert_gtest_xml.py --allow-skips`。不是 77 的非 0 仍让该步失败。smoke-8 仍在 `MORTRED_CI_REQUIRE_WEIGHTS=1` 下跑，缺权重仍是退出码 1。

这个二进制改为自己的 `main`，以便在 `RUN_ALL_TESTS` 返回后设置退出码。它不再链接 `gtest_main`。

代价：无权重时，直接跑 golden 不再是退出码 0；`scripts/run_tests.sh` 或 `ctest -R '^model_golden_test$'` 会把这一项显示为 Skipped。`cmake --build … --target check` 的总退出码不变。nightly 剩余 zoo 要多接住退出码 77，inventory 仍在。不增加进程，不读到权重字节，不重建 engine。

### 否决

把缺权重的默认从 `GTEST_SKIP` 改成 `GTEST_FAIL`，相当于本地也总是打开 `MORTRED_CI_REQUIRE_WEIGHTS`。cpu-profile 的 `check` 只拉了 hosted 子集，其余用例没有权重，整项会失败。未跑数值和数值不符都变成退出码 1。nightly 剩余 zoo 在 `set -e` 下会在 `--allow-skips` 之前停住。这和「分开两种结果」不是同一件事，也会改掉已经写明的 skip-as-pass。

用 CTest 的 `SKIP_REGULAR_EXPRESSION` 去匹配输出里任意 `SKIPPED`。`model_golden_test` 在 CTest 里是一整份二进制。hosted 权重已经比对通过、其余用例仍跳过时，这一整项会被标成 Skipped，cpu-profile 的 `check` 就不再把已经跑过的数值显示为 Passed。只改文档、让人自己翻 gtest 的 SKIPPED 行也不可用：`check` 使用 `--output-on-failure`，退出码 0 的那一项不会把 SKIPPED 正文打出来，结果行仍是 Passed。

## 改动面

- `test/model_golden_test.cc`：这个二进制的结束状态。文件头对「缺权重则整次算通过」的说明改成与退出码 77 一致。
- `test/CMakeLists.txt`：只给测试名 `model_golden_test` 设置 `SKIP_RETURN_CODE` 77；该目标链接 gtest 时不再链接 `gtest_main`。
- `test/ci_require_weights.h`：注释不再把缺权重写成「笔记本上的 golden 套件 stays green」。宏的两个分支保持不动。
- `docs/ci-golden-regression.md`：只改「本地无权重仍把 golden 当成通过」那一句，写明全部缺权重时退出码 77 与 `NUMERICAL_NOT_RUN`。
- `docs/model-developer-guide.md`：只改 “Missing weights → case skips” 那一句，口径与上一份相同。单条仍可以跳过。
- `.github/workflows/ci.yml`：只改 `gpu-nightly-full` 里 `unset MORTRED_CI_REQUIRE_WEIGHTS` 之后、写出 `gpu-rest.xml` 的那一次调用，使退出码 77 仍进入现有 `--allow-skips`。

## 明确不改

- `MORTRED_SKIP_OR_FAIL_WEIGHTS` 的分支：未要求权重时仍 `GTEST_SKIP`，要求权重时仍 `GTEST_FAIL`。
- `test/backend_unittest.cc` 以及其它非 golden 测试对这个宏的使用。缺权重时这些进程的退出码仍是 0。
- `weights_available` 的查找规则：仍看 toml 里的 `model_file_path` 与 `vocab_file_path`。
- `MORTRED_UPDATE_GOLDEN=1` 在权重存在并写出基线后的跳过：退出码仍是 0，文案仍含 `golden updated`。
- 同一进程里既有完成比对又有缺权重跳过时的退出码 0。cpu-profile 的 `check` 在 hosted 权重已拉下时，`model_golden_test` 仍是 Passed。
- `force_cpu_backend`，产品 toml 的 `type=mnn` / `type=tensorrt`，ONNX 与 LIBFACE 的现有约定。
- hosted、`gpu-pr-gate` smoke-8 的 fail-closed：仍是 `MORTRED_CI_REQUIRE_WEIGHTS=1` 加上 `scripts/ci_assert_gtest_xml.py`，且不允许 skip。
- `inference-paths`、`scripts/ci_eval_inference_paths.py`、`scripts/ci_print_tier_summary.py`。
- `scripts/ci_assert_gtest_xml.py` 的规则：有 skip 且未给 `--allow-skips` 时仍失败；给了该参数时仍写出 skip 名单。
- `scripts/run_tests.sh` 的参数和行为。它继续调用 ctest。
- gateway :8080、supervisor :8787、一 catalog 一进程、engine 的生成方式。
- 数值门限、golden 基线文件、用例名、`--gtest_filter` 的名字。

## 契约

没有新的 HTTP 路径、请求体、状态码或 toml 键。tensor 布局不变。

`model_golden_test` 增加两个进程级标记，各占标准输出一行，全文就是标记本身：

- `NUMERICAL_NOT_RUN`：本次选中用例全部因缺权重未做数值比对。进程退出码 77。
- `NUMERICAL_PASSED`：本次选中用例全部做完数值比对且失败数为 0。进程退出码 0。

两种标记不会出现在同一次运行里。失败、部分跳过、`golden updated`、以及没有选中用例时，两行都不出现。

`--gtest_output=xml:` 仍由 gtest 在进程退出前写完。全部缺权重时，xml 里对应条仍是 skipped；退出码 77 只表示这次进程没有做数值比对。CTest 看到 77 时，名为 `model_golden_test` 的那一项是 Skipped。该项为 Skipped 且没有其它失败时，ctest 自己的退出码是 0。

`MORTRED_CI_REQUIRE_WEIGHTS` 的取值规则不变。置位且缺权重时，进程退出码是 1，不是 77。

## 性能与资源

一个 catalog id 仍是一个 OS 进程。gateway 仍是 :8080，supervisor 仍是 :8787。结束状态在用例跑完之后计算，不多开进程，不加载权重，不分配推理显存，不调用 `mortredctl prepare`，不生成 engine。cpu profile 不因此链接 TensorRT。gpu profile 仍是 CUDA 12.x 与 TensorRT 10.x。

## 验收标准

下列命令的工作目录是仓库根。二进制是 cpu profile、`MORTRED_BUILD_FULL=ON` 的 `build-cpu/bin/model_golden_test` 与 `build-cpu/bin/backend_unittest`。`LD_LIBRARY_PATH` 包含 `build-cpu/lib` 与 `3rd_party/libs`。第 1 条到第 5 条的前提是文件 `weights/classification/mobilenetv2/mobilenetv2_ilsvrc2012.mnn` 不存在。

1. 未设置 `MORTRED_CI_REQUIRE_WEIGHTS`，未设置 `MORTRED_UPDATE_GOLDEN`，运行 `model_golden_test --gtest_filter=model_golden.mobilenetv2_classification`。进程退出码为 77。标准输出含一行 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。
2. 仅设置 `MORTRED_CI_REQUIRE_WEIGHTS=1`，同一过滤。进程退出码为 1。标准输出不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。
3. 仅设置 `MORTRED_UPDATE_GOLDEN=1`，同一过滤。进程退出码为 77。标准输出含 `NUMERICAL_NOT_RUN`，不含 `golden updated`，不含 `NUMERICAL_PASSED`。
4. 两个变量都未设置，`GTEST_FILTER=model_golden.mobilenetv2_classification`，执行 `ctest --test-dir build-cpu -R '^model_golden_test$'`。ctest 退出码为 0。输出里有 `model_golden_test` 且有 `Skipped`，该项结果不是 `Passed`。
5. 未设置 `MORTRED_CI_REQUIRE_WEIGHTS`，运行 `backend_unittest --gtest_filter=MnnSession.InitAndRunMobilenetv2`。进程退出码为 0。标准输出不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。
6. 运行 `model_golden_test --gtest_list_tests`。进程退出码为 0。标准输出不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。
7. 运行 `model_golden_test --gtest_filter=model_golden.does_not_exist`。进程退出码不是 77。标准输出不含这两个标记。
8. 两份文档分开判定。`docs/ci-golden-regression.md` 同时含 `NUMERICAL_NOT_RUN`、`77`，以及原文 `单条缺权重仍可以跳过` 或 `case skips` 二者之一。`docs/model-developer-guide.md` 同样同时含这三项。一份文档只含 `NUMERICAL_NOT_RUN` 和 `77`、没有上述单条跳过原文时，第 8 条不通过。`test/ci_require_weights.h` 不含 `stays green` 单独判定，不替代两份文档的判定。
9. 只看 `.github/workflows/ci.yml` 的 `gpu-nightly-full` 里步骤名含 `smoke-8` 的那一步。该步在第一次 `unset MORTRED_CI_REQUIRE_WEIGHTS` 之前仍含 `MORTRED_CI_REQUIRE_WEIGHTS=1`。从这句 `unset` 到该步第一次 `gpu-rest-skips.json` 之前，同时含 `77`、`gpu-rest.xml` 和 `--allow-skips`。同文件更早位置的 `MORTRED_CI_REQUIRE_WEIGHTS=1` 不能算这一步通过。

## 测试义务

黑盒：在没有上述 mnn 文件的仓库里执行「自测命令」。第 1 条到第 7 条看进程退出码和标准输出。第 4 条看 ctest 的退出码和结果文字。第 8 条对两份文档各自检索，并单独判定 `test/ci_require_weights.h` 不含 `stays green`。第 9 条只看 `gpu-nightly-full` 里步骤名含 `smoke-8` 的那一步。不启动 gateway 或 supervisor，不要求 GPU，不要求准备 engine。

白盒：宏的未要求权重分支仍是 `GTEST_SKIP`，要求权重分支仍是 `GTEST_FAIL`。`backend_unittest` 仍调用该宏。`NUMERICAL_PASSED` 只在失败数为 0 且每条选中用例都做完数值比对时打印。部分缺权重且失败数为 0 时退出码为 0，两个标记都不打印。权重存在时的 `golden updated` 全部跳过仍是退出码 0。`model_golden_test` 只保留一份 `main`。退出码 77 发生在 gtest 写完 xml 之后。

本机若已有该 mnn 文件，第 1 条到第 5 条不在这台机器上判定；不要把一次真实的数值通过记成第 1 条失败。

## 自测命令

前提与库路径：

```bash
test ! -f weights/classification/mobilenetv2/mobilenetv2_ilsvrc2012.mnn
export LD_LIBRARY_PATH="$PWD/build-cpu/lib:$PWD/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`test ! -f` 的期望退出码是 0。其后命令都在这个前提下执行。

```bash
env -u MORTRED_CI_REQUIRE_WEIGHTS -u MORTRED_UPDATE_GOLDEN \
  ./build-cpu/bin/model_golden_test --gtest_filter=model_golden.mobilenetv2_classification
```

期望退出码 77。标准输出含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。

```bash
env -u MORTRED_UPDATE_GOLDEN MORTRED_CI_REQUIRE_WEIGHTS=1 \
  ./build-cpu/bin/model_golden_test --gtest_filter=model_golden.mobilenetv2_classification
```

期望退出码 1。标准输出不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。

```bash
env -u MORTRED_CI_REQUIRE_WEIGHTS MORTRED_UPDATE_GOLDEN=1 \
  ./build-cpu/bin/model_golden_test --gtest_filter=model_golden.mobilenetv2_classification
```

期望退出码 77。标准输出含 `NUMERICAL_NOT_RUN`，不含 `golden updated`，不含 `NUMERICAL_PASSED`。

```bash
env -u MORTRED_CI_REQUIRE_WEIGHTS -u MORTRED_UPDATE_GOLDEN \
  GTEST_FILTER=model_golden.mobilenetv2_classification \
  ctest --test-dir build-cpu -R '^model_golden_test$'
```

期望退出码 0。输出含 `Skipped` 与 `model_golden_test`，该项不是 `Passed`。

```bash
env -u MORTRED_CI_REQUIRE_WEIGHTS \
  ./build-cpu/bin/backend_unittest --gtest_filter=MnnSession.InitAndRunMobilenetv2
```

期望退出码 0。标准输出不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。

```bash
./build-cpu/bin/model_golden_test --gtest_list_tests
```

期望退出码 0。标准输出不含这两个标记。

```bash
set +e
./build-cpu/bin/model_golden_test --gtest_filter=model_golden.does_not_exist > /tmp/golden-none.out
code=$?
set -e
test "$code" -ne 77
```

`test` 的期望退出码是 0。`/tmp/golden-none.out` 不含 `NUMERICAL_NOT_RUN`，不含 `NUMERICAL_PASSED`。

```bash
rg -n -F "NUMERICAL_NOT_RUN" docs/ci-golden-regression.md \
  && rg -n -F "77" docs/ci-golden-regression.md \
  && rg -n -F -e "单条缺权重仍可以跳过" -e "case skips" docs/ci-golden-regression.md
```

期望退出码 0。只检索这一份文档。没有「单条缺权重仍可以跳过」或 `case skips` 时，退出码不是 0。

```bash
rg -n -F "NUMERICAL_NOT_RUN" docs/model-developer-guide.md \
  && rg -n -F "77" docs/model-developer-guide.md \
  && rg -n -F -e "单条缺权重仍可以跳过" -e "case skips" docs/model-developer-guide.md
```

期望退出码 0。只检索这一份文档。没有「单条缺权重仍可以跳过」或 `case skips` 时，退出码不是 0。

```bash
rg -n -F "stays green" test/ci_require_weights.h
```

期望退出码 1。这一条单独判定，不代替上面两份文档。

```bash
python3 - <<'PY'
from pathlib import Path
lines = Path(".github/workflows/ci.yml").read_text(encoding="utf-8").splitlines()
start = None
for i, line in enumerate(lines):
    if line.startswith("  gpu-nightly-full:") and not line.startswith("   "):
        start = i
        break
assert start is not None
end = len(lines)
for i in range(start + 1, len(lines)):
    raw = lines[i]
    if (
        raw.startswith("  ")
        and not raw.startswith("   ")
        and not raw.strip().startswith("#")
        and raw.strip().endswith(":")
    ):
        end = i
        break
job = lines[start:end]
step_i = None
for i, line in enumerate(job):
    if line.startswith("      - name:") and "smoke-8" in line:
        step_i = i
        break
assert step_i is not None
step_end = len(job)
for i in range(step_i + 1, len(job)):
    if job[i].startswith("      - name:"):
        step_end = i
        break
step = "\n".join(job[step_i:step_end])
unset = "unset MORTRED_CI_REQUIRE_WEIGHTS"
u = step.find(unset)
assert u != -1
assert "MORTRED_CI_REQUIRE_WEIGHTS=1" in step[:u]
j = step.find("gpu-rest-skips.json", u)
assert j != -1
mid = step[u:j]
assert "77" in mid and "gpu-rest.xml" in mid and "--allow-skips" in mid
PY
```

期望退出码 0。脚本只取 `gpu-nightly-full` 内第一个步骤名含 `smoke-8` 的步骤。`MORTRED_CI_REQUIRE_WEIGHTS=1` 只在该步第一次 `unset MORTRED_CI_REQUIRE_WEIGHTS` 之前查找。`77`、`gpu-rest.xml`、`--allow-skips` 只在这句 `unset` 到该步第一次 `gpu-rest-skips.json` 之前查找。同文件更早位置的 `MORTRED_CI_REQUIRE_WEIGHTS=1` 不会使退出码变成 0。

## RD 不得自行决定

- 不得把「全部缺权重」的退出码定成 0 或 1。77 只表示未跑数值。1 只表示失败，包括 `MORTRED_CI_REQUIRE_WEIGHTS=1` 时的缺权重。
- 不得新增环境变量，把全部缺权重恢复成退出码 0。
- 不得改 `MORTRED_SKIP_OR_FAIL_WEIGHTS` 的分支，不得改 `backend_unittest` 的退出码。
- 缺权重跳过的原文保持 `weights not available`。更新基线的原文保持 `golden updated`。
- 不得在部分用例已经完成数值比对时，把整个 `model_golden_test` 标成 Skipped 或让 `check` 失败。
- 不得在 nightly 剩余 zoo 得到 77 时跳过 `--allow-skips`，也不得把不是 77 的失败放行。
- 不得改 smoke-8、hosted golden、`inference-paths` 的 fail-closed。
- 不得改 `force_cpu_backend`、用例名、数值门限、产品 toml、HTTP 契约。
- 两个标记的正文就是 `NUMERICAL_NOT_RUN` 与 `NUMERICAL_PASSED`，各一行，写到标准输出。
