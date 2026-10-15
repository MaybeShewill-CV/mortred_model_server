# 技术方案

- 问题: P2-21
- 版本: v1
- 对应列表: `cowork/ledger/round-002/02-p0-p1-final.md` 锁定队列 v3 的 P2-21
- 基线: P2-20 已合入（merge `bb1afafb952a14f18e0c996395cf46ae48f87215`）。`--product-trt` 默认 `no`；工作流显式传 `--product-trt no`。本条在其上补 XML 核验。

## 边界

本方案只修一件事：当 `inference-paths` 打印 `TRT_PROOF=yes` 时，必须以同一次 `gpu-pr-gate` 的 gtest XML 为据，确认至少有一个产品 `type=tensorrt` 用例且该用例未被 skip（`skipped=0`）。没有这份证据时不得打印 `TRT_PROOF=yes`。

锁定的可观察变化（队列 v3）：日志出现 `TRT_PROOF=yes` 时，同一次 `gpu-pr-gate` 的 gtest XML 里至少有一个产品 `type=tensorrt` 用例，且 `skipped=0`。看这一次日志和这一份 XML 即可。

出处与现状（合入 P2-20 后）：`evaluate(..., product_trt="yes")` 在 `gpu == success` 时直接打印 `TRT_PROOF=yes`，不读 gtest XML。产品 `type=tensorrt` 用例（gtest suite `Yolov8TrtDetection`、`TrtProductGolden`）不在 `MORTRED_GPU_SMOKE_FILTER` 里。现行工作流传 `--product-trt no`，主线不会出现 `TRT_PROOF=yes`；但一旦误传 `--product-trt yes` 或未来接线，仍可出现假阳性。`gpu-pr-gate` 已写出 `${RUNNER_TEMP}/gtest/gpu-smoke.xml` 并上传 artifact。

与 P2-20 的分工（必须守住）：

- P2-20：关掉「纯 ONNX smoke 成功仍打印 `TRT_PROOF=yes`」。
- 本条（P2-21）：凡打印 `TRT_PROOF=yes`，必须以 XML 证明同一次跑里有产品 `type=tensorrt` 且未 skip。不扩大 `MORTRED_GPU_SMOKE_FILTER`。

不升为 P0/P1。不改模型进程、TensorRT 运行时、engine 生成、worker 池或共享 context。

## 做法

### 推荐

做法一：给 `ci_eval_inference_paths.py` 增加 `--gtest-xml PATH`。只有 `gpu == success` 且该 XML 证明「至少一个产品 `type=tensorrt` 用例且未 skip」时，才打印 `TRT_PROOF=yes`。`--product-trt yes`  alone 不再足够。

1. 新增 CLI `--gtest-xml`（可选路径）。语义：同一次 `gpu-pr-gate` 写出的 gtest XML（现行为 `gpu-smoke.xml`）。
2. 新增纯函数 `xml_has_ran_product_trt(path) -> bool`：
   - 解析 gtest XML（复用或镜像 `ci_assert_gtest_xml.parse_gtest_xml` 的遍历方式；允许直接 `xml.etree.ElementTree`）。
   - 产品 `type=tensorrt` 用例的判定：testcase 所属 suite 名为 `Yolov8TrtDetection` 或 `TrtProductGolden`（与 `test/model_golden_test.cc` 一致）。
   - 「跑过且 skipped=0」：该 testcase 不是 skipped（无 `<skipped>` 子节点，`result` 不是 `skipped`，`status` 不是 `notrun`）。
   - 至少命中一条即 True；文件缺失或 XML 非法 → False（调用方决定文案）。
3. 改 `evaluate(cpu, gpu, product_trt="no", gtest_xml=None)` 在 `gpu == success` 分支：
   - 仅当 `xml_has_ran_product_trt(gtest_xml)` 为 True 时打印 `TRT_PROOF=yes` 并返回 0。
   - 否则打印 `TRT_PROOF=no` 并返回 0（含：未传 XML；XML 只有 ONNX/`model_golden`；产品 suite 全是 skip；`--product-trt yes` 但无 XML 证明）。
   - **不再**仅凭 `product_trt == "yes"` 打印 `TRT_PROOF=yes`。`--product-trt` 保留：`no` 时即使误传了无关 XML 也不改变「现行工作流不声称 yes」的意图——进一步约束：打印 `TRT_PROOF=yes` 还要求调用方显式 `--product-trt yes` **且** XML 证明。这样现行 `--product-trt no` 工作流即使以后下载了 XML 也不会突然变 yes；未来接线需同时改 flag 与过滤器/XML。
   - 最终规则：`TRT_PROOF=yes` ⟺ `gpu == success` ∧ `product_trt == "yes"` ∧ XML 证明产品 TRT 已跑且未 skip。
4. 更新 `--self-test`：
   - 用临时 fixture XML：仅 `model_golden.*` → 即使 `product_trt=yes` 也得 `TRT_PROOF=no`。
   - fixture 含 `TrtProductGolden.mobilenetv2` 未 skip → `product_trt=yes` + 该 XML → `TRT_PROOF=yes`。
   - 同名 suite 但 testcase 为 skipped → `TRT_PROOF=no`。
   - 不传 XML + `product_trt=yes` → `TRT_PROOF=no`（堵住 P2-20 残留假阳性）。
   - 默认 / `product_trt=no` 行为与 P2-20 一致。
   - 跳过路径不变。
5. `check_workflow`：继续要求 `--cpu` / `--gpu`；**不**要求现在就传 `--gtest-xml` 或改成 `--product-trt yes`（过滤器仍是 ONNX）。允许继续 `--product-trt no`。
6. 文档：`docs/ci-golden-regression.md` 写明 `TRT_PROOF=yes` 需要 `--product-trt yes` **且** 同一次 gtest XML 含未 skip 的产品 `type=tensorrt` suite（`Yolov8TrtDetection` / `TrtProductGolden`）。

代价：在工作流把 smoke 接到产品 TRT 并传 XML + `--product-trt yes` 之前，主线仍只有 `TRT_PROOF=no`。这符合 P2-20/P2-21 边界。无 engine/显存代价。

### 否决

做法二：扩大 `MORTRED_GPU_SMOKE_FILTER` 纳入 `TrtProductGolden` / `Yolov8TrtDetection`。队列写明过滤器不扩成全目录。否决。

做法三：只改文档、工作流仍可 `--product-trt yes` 无 XML。否决。

做法四：`product_trt=yes` 但 XML 不达标时改 exit 1。会把「配置撒谎」与「GPU 失败」混在 exit 1；锁定句只要求日志不出现假 `TRT_PROOF=yes`。用 `TRT_PROOF=no` + exit 0 即可。否决 exit 1。

做法五：仅信任 `--product-trt yes`、不读 XML。那是 P2-20 留下的洞。否决。

## 改动面

- `scripts/ci_eval_inference_paths.py`：`--gtest-xml`；`xml_has_ran_product_trt`；`evaluate` / `self_test` / 文档字符串。
- `docs/ci-golden-regression.md`：一句 XML 核验条件。
- 不改 `.github/workflows/ci.yml` 的过滤器与现行 `--product-trt no`（可选：以后接线再下 XML；本条不强制）。

## 明确不改

- `MORTRED_GPU_SMOKE_FILTER`；`conf/ci_hosted_golden.json` 的 `gpu_smoke.cases`。
- `gpu-pr-gate` 构建、转换、过滤器名单、XML 上传路径名（仍可读其产物做自测 fixture）。
- 不把产品 TRT 扩进 PR smoke。
- UX3 跳过路径；`cpu-profile`。
- 模型进程、gateway、supervisor、engine、static shape、batch=1、独立 context。

## 契约

HTTP / OpenAPI / 产品 toml 不变。CI 评判：`TRT_PROOF=yes` 仅在 `product_trt=yes` 且 XML 证明产品 TRT 未 skip 时出现；否则 success 路径为 `TRT_PROOF=no` + exit 0。

## 性能与资源

只跑 Python 自测与只读文档。不生成 engine，不拉起模型。

## 验收标准

1. `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success --product-trt yes`（不传 `--gtest-xml`）。退出码 0。stdout 含 `TRT_PROOF=no`，不含 `TRT_PROOF=yes`。
2. 构造仅含 `model_golden` 未 skip 的 XML，跑 `--product-trt yes --gtest-xml <该文件>`。退出码 0。`TRT_PROOF=no`，无 `TRT_PROOF=yes`。
3. 构造含 `TrtProductGolden.mobilenetv2` 未 skip 的 XML，跑 `--product-trt yes --gtest-xml <该文件>`。退出码 0。`TRT_PROOF=yes`，无 `TRT_PROOF=no`。
4. 同 suite 但该 case 为 skipped 的 XML + `--product-trt yes` → `TRT_PROOF=no`。
5. 默认 / `--product-trt no` + 任意 XML → 仍为 `TRT_PROOF=no`（不因 XML 误升 yes）。
6. `--cpu success --gpu skipped` 仍 exit 2 + `NOT_A_TRT_PROOF`。
7. `python3 scripts/ci_eval_inference_paths.py --self-test` 退出码 0。
8. diff 不扩大 `MORTRED_GPU_SMOKE_FILTER`；工作流可继续 `--product-trt no`。

判定：本机 WSL Ubuntu、仓库或 worktree 根即可。不要求真实 GPU runner。

## 测试义务

QA 在本机 WSL Ubuntu 跑验收 1–7，记录退出码与关键 stdout；`git show --stat` 确认改动面；打开文档核对。

## 自测命令

```bash
cd /mnt/g/Codex/mortred_wt_r2_p2_21
python3 scripts/ci_eval_inference_paths.py --self-test
python3 - <<'PY'
import subprocess, sys, tempfile
from pathlib import Path
script = "scripts/ci_eval_inference_paths.py"

def run(args):
    p = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

code, out, err = run(["--cpu", "success", "--gpu", "success", "--product-trt", "yes"])
assert code == 0 and "TRT_PROOF=no" in out and "TRT_PROOF=yes" not in out, (code, out, err)

onnx = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8")
onnx.write("""<?xml version='1.0'?>
<testsuites tests='1' failures='0' disabled='0' skipped='0'>
  <testsuite name='model_golden' tests='1' failures='0' skipped='0'>
    <testcase name='yolov8_detection' status='run' result='completed'/>
  </testsuite>
</testsuites>
""")
onnx.close()
code, out, err = run(["--cpu", "success", "--gpu", "success", "--product-trt", "yes", "--gtest-xml", onnx.name])
assert code == 0 and "TRT_PROOF=no" in out and "TRT_PROOF=yes" not in out, (code, out, err)

trt = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8")
trt.write("""<?xml version='1.0'?>
<testsuites tests='1' failures='0' disabled='0' skipped='0'>
  <testsuite name='TrtProductGolden' tests='1' failures='0' skipped='0'>
    <testcase name='mobilenetv2' status='run' result='completed'/>
  </testsuite>
</testsuites>
""")
trt.close()
code, out, err = run(["--cpu", "success", "--gpu", "success", "--product-trt", "yes", "--gtest-xml", trt.name])
assert code == 0 and "TRT_PROOF=yes" in out and "TRT_PROOF=no" not in out, (code, out, err)

code, out, err = run(["--cpu", "success", "--gpu", "success", "--product-trt", "no", "--gtest-xml", trt.name])
assert code == 0 and "TRT_PROOF=no" in out and "TRT_PROOF=yes" not in out, (code, out, err)

print("p2-21-trt-xml ok")
PY
```

期望整段退出码 0。
