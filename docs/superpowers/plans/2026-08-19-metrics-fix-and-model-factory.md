# 指标顺序修复 + 模型工厂去反模式实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复复评遗留清单 #1（`base_server_impl.h:655` 推理耗时直方图恒记 0 的顺序 bug）与 #3（28 个模型工厂函数"每次 create 都写全局注册表"的反模式），全部以 TDD/结构门禁驱动，行为等价由 golden 回归锚定。

**架构：** 任务 1 在 e2e 契约测试中新增"请求后 `/metrics` 的 `inference_duration_ms_sum` 必须 > 0"断言（当前红），交换两行修复（绿）。任务 2 把 10 个工厂头的 28 个模型 create 函数从"register_type + create"改为直接构造（签名不变、调用方零改动），并在 `check_consistency.py` 加结构门禁"`src/factory/*_task.h` 禁止出现 `register_type<`"（当前红，迁移后绿）。

**技术栈：** C++17、GTest、Python 一致性脚本、WSL 构建环境（`LD_LIBRARY_PATH=3rd_party/libs:_lib`）。

---

### 已核实的事实（2026-08-19，main@3b3e546）

1. bug 现场 `base_server_impl.h`（do_work 尾部）：

```cpp
_m_metrics.observe_inference_duration_ms(result->worker_run_time_consuming);  // 观测旧值（恒 0）
result->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000; // 赋值在后
```

   对照组 `queue_wait`（L627-628）顺序正确（先赋值后观测），证明这是孤立的复制顺序错误。
2. `ModelFactory<Base>` 的消费者**只有** 10 个 `src/factory/*_task.h` 自身（全仓 grep 核实）；无任何外部按名查找模型的代码 → 直接构造安全。
3. 28 个模型 create 函数全部同构（仅参数名差异：detector_name/model_name/...），分布在 classification(4)/clip(1)/enhancement(3)/feature_point(1)/matting(2)/mono_depth(2)/obj_detection(7)/ocr(1)/sam(3)/scene_seg(4)。
4. 模型构造不接收 name（`register_type` 闭包即 `new CONCRETE()`），name 仅作注册表键 → 直接构造后 name 无行为影响，保留参数以维持 API 兼容。

---

### 任务 1：推理耗时直方图恒 0 的 TDD 修复

**文件：**
- 修改：`test/server_e2e_contract_test.cc`（追加用例）
- 修改：`src/server/base_server_impl.h:655-656`（交换两行）

- [ ] **步骤 1：编写失败测试**

在 `test/server_e2e_contract_test.cc` 的 e2e 用例区追加（复用现有 `start_server`/`send_request` 夹具）：

```cpp
TEST(server_e2e_contract, metrics_inference_duration_sum_is_positive_after_request) {
    ServerHandle handle = start_server();
    const std::string body = "{\"img_data\":\"aGVsbG8=\"}";
    auto resp = send_request(handle.port, "POST", "/test/model", body, k_json_auth_headers);
    ASSERT_EQ(resp.status, 200);

    auto metrics = send_request(handle.port, "GET", "/metrics");
    ASSERT_EQ(metrics.status, 200);
    const std::string key = "mortred_inference_duration_ms_sum";
    const auto pos = metrics.body.find(key);
    ASSERT_NE(pos, std::string::npos) << metrics.body;
    const auto value_pos = metrics.body.find(' ', pos + key.size());
    ASSERT_NE(value_pos, std::string::npos);
    const double sum = std::atof(
        metrics.body.substr(value_pos + 1, metrics.body.find('\n', value_pos)).c_str());
    EXPECT_GT(sum, 0.0) << "inference duration histogram must observe the real "
                        << "run time, not the pre-assignment zero";
}
```

- [ ] **步骤 2：运行验证失败（红）**

```bash
cd build/tests-only && LD_LIBRARY_PATH=../../3rd_party/libs:../../_lib \
  ctest -R server_e2e_contract_test --output-on-failure
```

预期：新用例 FAIL（sum == 0），其余 11 个用例 PASS。

- [ ] **步骤 3：交换两行修复**

```cpp
    auto task_finish_ts = Timestamp::now();
    result->task_finished_ts = task_finish_ts.to_format_str();
    result->worker_run_time_consuming = (task_finish_ts - task_receive_ts) * 1000;
    _m_metrics.observe_inference_duration_ms(result->worker_run_time_consuming);
```

- [ ] **步骤 4：运行验证通过（绿）**

同步骤 2 命令；预期 12/12 PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/server/base_server_impl.h test/server_e2e_contract_test.cc
git commit -m "fix(server): observe inference duration after computing it (遗留 #1)"
```

---

### 任务 2：模型工厂直接构造化（28 个函数）

**文件：**
- 修改：`scripts/check_consistency.py`（新增结构门禁，先行红）
- 修改：`src/factory/{10 个 *_task.h}`（28 个函数体替换）
- 不改：任何调用方（签名不变）；`base_factory.h` 的 `ModelFactory` 别名与工厂单测保留（`TypeErasedFactory` 仍由 server 侧使用）

- [ ] **步骤 1：结构门禁先行（红）**

`check_consistency.py` 追加：

```python
def check_factory_register_type_banned() -> list[str]:
    """src/factory/*_task.h 禁止 register_type：模型 create 一律直接构造，
    服务 create 用 register_creator 闭包（消除每次创建写全局注册表）。"""
    errors: list[str] = []
    for header in sorted((ROOT / "src" / "factory").glob("*_task.h")):
        for i, line in enumerate(header.read_text(encoding="utf-8").splitlines(), 1):
            if "register_type" in line:
                errors.append(
                    f"{header.relative_to(ROOT)}:{i}: register_type is banned in task "
                    f"headers (models construct directly; servers use register_creator)")
    return errors
```

并注册进主检查列表。运行 `python3 scripts/check_consistency.py` 预期 **FAIL（28 处）**。

- [ ] **步骤 2：脚本迁移 28 个函数**

模式（三行体 → 直接构造，参数名保留并 `(void)` 消未用告警）：

```cpp
// 迁移前
auto& model_factory = ModelFactory<BaseAiModel<INPUT, OUTPUT> >::get_instance();
model_factory.template register_type<Concrete<INPUT, OUTPUT> >(xxx_name);
return model_factory.create(xxx_name);

// 迁移后
// 直接构造：模型创建不写全局注册表（无副作用、无互斥开销），
// 消除"每次 create 都 register"反模式；name 仅保留以兼容调用方
(void)xxx_name;
return std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new Concrete<INPUT, OUTPUT>());
```

用 Python 正则 `\{\s*auto&\s+model_factory.*?register_type<\s*(\w+)<INPUT,\s*OUTPUT>\s*>\((\w+)\);\s*return\s+model_factory\.create\(\2\);\s*\}` 批量替换，脚本断言恰好命中 28 处。

- [ ] **步骤 3：结构门禁转绿**

```bash
python3 scripts/check_consistency.py   # 预期 PASS
```

- [ ] **步骤 4：行为等价验证（黄金锚点）**

```bash
cmake --build build/full -j16
ctest --test-dir build/tests-only 全绿（21 项）
LD_LIBRARY_PATH=_lib:3rd_party/libs _bin/model_golden_test   # 21/21（工厂路径未变行为）
GLOG_logtostderr=1 _bin/mobilenetv2_benchmark.out \
  ../conf/model/classification/mobilenetv2/mobilenetv2_config.toml   # 分类结果 id=170
```

- [ ] **步骤 5：Commit**

```bash
git add scripts/check_consistency.py src/factory
git commit -m "refactor(factory): construct models directly, ban register_type (遗留 #3)"
```

---

## 自检

1. **覆盖度**：#1 = 任务 1（红绿 + e2e 断言）；#3 = 任务 2（结构门禁红绿 + 28 函数迁移 + golden 行为锚定）✓
2. **占位符**：无；所有代码/命令可直接执行 ✓
3. **类型一致性**：`std::unique_ptr<BaseAiModel<INPUT, OUTPUT> >(new Concrete<INPUT, OUTPUT>())` 与原 `register_type` 闭包 `new CONCRETE()` 同型；签名/命名空间不变 ✓
