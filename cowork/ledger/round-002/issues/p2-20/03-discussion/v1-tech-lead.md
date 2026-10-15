# 技术方案

- 问题: P2-20
- 版本: v1
- 对应列表: `cowork/ledger/round-002/02-p0-p1-final.md` 锁定队列 v3 的 P2-20

## 边界

本方案只修一件事：当 `gpu-pr-gate` 结果为 `success`、但这次作业实际跑的是现行四个 `type=onnx` golden（`MORTRED_GPU_SMOKE_FILTER` 指向的 hosted/cpu ONNX 配置）时，`inference-paths` 不得打印 `TRT_PROOF=yes`。

锁定的可观察变化（队列 v3）：只跑下面四个用例，且 yolov8 引擎步骤被 skip 或只写出引擎文件时，看这一次 `inference-paths` 日志，其中没有 `TRT_PROOF=yes`。四个名字是 `model_golden.yolov8_detection`、`model_golden.yolov8_mixed_size_batch_matches_single_runs`、`model_golden.mobilenetv2_classification`、`model_golden.libface_detection`。它们加载的配置 `backend.type` 都是 `onnx`。

出处与现状：`scripts/ci_eval_inference_paths.py` 的 `evaluate` 在 `cpu == success` 且 `gpu == success` 时无条件打印 `TRT_PROOF=yes` 并返回 0。`.github/workflows/ci.yml` 的 `inference-paths` 只传入 `--cpu` / `--gpu` 两个 needs 结果。`gpu-pr-gate` 的 `--gtest_filter` 是上述四个 ONNX 用例；产品 `type=tensorrt` 用例（如 `Yolov8TrtDetection.*`、`TrtProductGolden.*`）不在过滤器里。

与 P2-21 的分工（必须守住）：

- 本条（P2-20）关掉「纯 ONNX smoke 成功仍打印 `TRT_PROOF=yes`」这条假阳性。
- P2-21 负责：当日志出现 `TRT_PROOF=yes` 时，核对同一次 `gpu-pr-gate` 的 gtest XML 里至少有一个产品 `type=tensorrt` 用例且 `skipped=0`。本条不读 gtest XML，不把 XML 核验写进本条验收。

不升为 P0/P1。不把 `MORTRED_GPU_SMOKE_FILTER` 扩成全目录。不改模型进程、TensorRT 运行时、engine 生成算法、worker 池或共享 context。

## 做法

### 推荐

做法一：给 `ci_eval_inference_paths.py` 增加显式的「本 job 是否跑过产品 `type=tensorrt` 用例」开关；只有开关为是时，`gpu == success` 才打印 `TRT_PROOF=yes`。现行工作流不传该开关（视为否），因此纯 ONNX smoke 成功时日志不再出现 `TRT_PROOF=yes`。

1. 新增 CLI 参数 `--product-trt`，取值 `yes` 或 `no`，默认 `no`。语义：`yes` 表示这次被评判的 GPU 作业实际跑过至少一个产品 `type=tensorrt` 用例；`no` 表示没有（含现行四个 ONNX smoke）。
2. 改 `evaluate(cpu, gpu, product_trt="no")`：
   - `cpu != success`：保持现状（stderr 报 cpu-profile，返回 1）。
   - `gpu == skipped`：保持现状（打印 `NOT_A_TRT_PROOF` 与免责声明，返回 2）。这是 UX3 跳过路径，本条不改。
   - `gpu` 为 failure / cancelled / 其他：保持现状（stderr 报 ended as，返回 1）。
   - `gpu == success` 且 `product_trt == "yes"`：打印 `TRT_PROOF=yes`，返回 0。
   - `gpu == success` 且 `product_trt != "yes"`（含默认 `no`）：**不**打印 `TRT_PROOF=yes`；打印一行 `TRT_PROOF=no`；返回 0。不把成功的 ONNX smoke 打成 exit 2，避免与「作业被跳过」混淆。
3. 更新 `--self-test`：
   - `("success","success")` 在默认 / `product_trt=no` 下期望退出码 0，stdout 含 `TRT_PROOF=no`，不含 `TRT_PROOF=yes`。
   - 增加 `product_trt=yes` 时 `("success","success")` 期望退出码 0，stdout 含 `TRT_PROOF=yes`，不含 `TRT_PROOF=no` / `NOT_A_TRT_PROOF`。
   - 跳过路径 `("success","skipped",2)` 的断言保持不变。
4. `check_workflow`：继续要求 `inference-paths` 调用本脚本并传入 `--cpu` / `--gpu`；**不**要求工作流现在就传 `--product-trt yes`（现行过滤器仍是 ONNX）。允许工作流显式传 `--product-trt no`，或不传（默认 no）。不在本条把 gtest XML 路径接进 `inference-paths`。
5. 文档：在 `docs/ci-golden-regression.md` 用一两句写明：`TRT_PROOF=yes` 仅当 `gpu-pr-gate` 成功且本脚本收到产品 `type=tensorrt` 已跑的肯定信号；现行 PR smoke 四个 ONNX 用例成功时打印 `TRT_PROOF=no`。不扩写过滤器名单。

代价：在 P2-21 合入之前，主线 `inference-paths` 在 GPU smoke 成功时也不会再出现 `TRT_PROOF=yes`。这正是本条锁定句要求的。没有 engine、显存或共享 context 代价。

### 否决

做法二：把 `Yolov8TrtDetection` / `TrtProductGolden.*` 扩进 `MORTRED_GPU_SMOKE_FILTER`，继续用「gpu success ⇒ TRT_PROOF=yes」。队列写明过滤器不扩成全目录；且「日志出现 yes 时 XML 里确有产品 tensorrt」是 P2-21 的验收。否决。

做法三：在本条就让 `evaluate` 读取 `gpu-smoke.xml` 并扫描产品 `type=tensorrt`。那是 P2-21 的核验路径。否决。

做法四：`gpu == success` 时改为打印 `NOT_A_TRT_PROOF` 并返回 2。会把「跑过且通过的 ONNX smoke」与「GPU 作业被跳过」混成同一退出码。否决。

做法五：只改文档、不改脚本。锁定验收读的是 `inference-paths` 日志里有没有 `TRT_PROOF=yes`。否决。

## 改动面

- `scripts/ci_eval_inference_paths.py`：新增 `--product-trt`；`evaluate` / `self_test` / 文档字符串按推荐改写。
- `docs/ci-golden-regression.md`：补一句 `TRT_PROOF=yes` / `TRT_PROOF=no` 与现行 ONNX smoke 的关系。
- 可选：`.github/workflows/ci.yml` 的 `inference-paths` 步骤显式加上 `--product-trt no`（与默认等价，仅使意图可见）。不加也可以，默认已是否。

## 明确不改

- `MORTRED_GPU_SMOKE_FILTER` 的四个名字与顺序；`conf/ci_hosted_golden.json` 的 `gpu_smoke.cases`。
- `gpu-pr-gate` 的构建、`convert_trt_engines.sh --only yolov8`、gtest 过滤器、XML 上传。
- 不读、不解析 gtest XML（留给 P2-21）。
- 不新增产品 `type=tensorrt` golden 进 PR smoke。
- `cpu-profile`、UX3 跳过路径（`NOT_A_TRT_PROOF` + exit 2）。
- 模型进程、gateway、supervisor、engine、static shape、batch=1、每个 worker 独立 TensorRT context。
- P2-21 的开放陈述与验收。

## 契约

HTTP / OpenAPI / 产品 toml 不变。变化只在 CI 评判文案：`gpu-pr-gate=success` 且未声明产品 tensorrt 已跑时，stdout 出现 `TRT_PROOF=no` 而非 `TRT_PROOF=yes`；退出码仍为 0。`gpu-pr-gate=skipped` 仍为 `NOT_A_TRT_PROOF` + exit 2。

## 性能与资源

不生成 engine，不改 static shape、batch=1，不改 `worker_nums`，不拉起模型。每个 worker 仍各自持有 TensorRT execution context。本条只跑 Python 自测与只读文档断言。

## 验收标准

1. 在仓库根执行 `python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success`（不传 `--product-trt`，或显式 `--product-trt no`）。退出码 0。stdout 含 `TRT_PROOF=no`。stdout 不含 `TRT_PROOF=yes`。
2. 同一次：`python3 scripts/ci_eval_inference_paths.py --cpu success --gpu success --product-trt yes`。退出码 0。stdout 含 `TRT_PROOF=yes`。stdout 不含 `TRT_PROOF=no`。
3. UX3 跳过路径不变：`--cpu success --gpu skipped` 退出码 2，stdout 含 `NOT_A_TRT_PROOF`，不含 `TRT_PROOF=yes`。
4. `python3 scripts/ci_eval_inference_paths.py --self-test` 退出码 0。
5. 本条 diff 不扩大 `MORTRED_GPU_SMOKE_FILTER`，不新增对 gtest XML 的读取；不把 P2-21 的 XML 核验写进本脚本的默认成功路径。
6. `docs/ci-golden-regression.md` 写明现行四个 ONNX smoke 成功时不是 `TRT_PROOF=yes`。

判定：以上命令在本机 WSL Ubuntu、仓库路径 `/mnt/g/Codex/mortred_model_server`（或 worktree 根）执行即可。不要求真实 GPU runner，不跑 `model_golden_test`，不跑 calibrate / 官方 RPS。

## 测试义务

QA 在本机 WSL Ubuntu：

1. 跑验收标准 1–4 的命令，记录退出码与关键 stdout 行。
2. 用 `git diff` / `git show --stat` 确认改动面落在设计所列文件；过滤器字符串与 gtest XML 解析未出现。
3. 打开 `docs/ci-golden-regression.md` 核对标准 6。
4. 不把本条升为 P0/P1；不要求拉起 self-hosted GPU。

## 自测命令

在本机 WSL Ubuntu 执行。工作目录为仓库根（基线或 worktree）。期望整段退出码为 0。

```bash
cd /mnt/g/Codex/mortred_model_server
python3 - <<'PY'
import subprocess, sys
script = "scripts/ci_eval_inference_paths.py"

def run(args):
    p = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

code, out, err = run(["--cpu", "success", "--gpu", "success"])
assert code == 0, (code, out, err)
assert "TRT_PROOF=no" in out, out
assert "TRT_PROOF=yes" not in out, out

code, out, err = run(["--cpu", "success", "--gpu", "success", "--product-trt", "yes"])
assert code == 0, (code, out, err)
assert "TRT_PROOF=yes" in out, out
assert "TRT_PROOF=no" not in out, out

code, out, err = run(["--cpu", "success", "--gpu", "skipped"])
assert code == 2, (code, out, err)
assert "NOT_A_TRT_PROOF" in out, out
assert "TRT_PROOF=yes" not in out, out

code, out, err = run(["--self-test"])
assert code == 0, (code, out, err)
print("p2-20-trt-proof ok")
PY
```

## RD 不得自行决定

- 不得在本条读取或解析 gtest XML（那是 P2-21）。
- 不得把产品 `type=tensorrt` 用例扩进 `MORTRED_GPU_SMOKE_FILTER`。
- 不得把 `gpu == success` 且 `product_trt=no` 改成 exit 2 或打印 `NOT_A_TRT_PROOF` 冒充跳过路径。
- 不得删除 `TRT_PROOF=yes` 这条正路径（须在 `--product-trt yes` 时仍可用，供 P2-21 接线）。
- 不得改动 gateway / 模型进程 / worker TensorRT context 结构。
